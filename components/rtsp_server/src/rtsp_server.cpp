/*
 * rtsp_server backed by cvi_rtsp.
 *
 * The only file in this component that knows which streaming library is in
 * use. Behaviour is a faithful port of the eight identical rtsp_demo.c copies
 * this replaces; everything that used to be a hard-coded constant there is now
 * a config field, and the resulting values are queryable so ONVIF can describe
 * the stream instead of guessing.
 *
 * Note on authentication: cvi_rtsp's C API has no notion of credentials, but
 * CVI_RTSP_CTX exposes the underlying live555 RTSPServer as a void*, so the
 * auth database is installed directly on it. sscma-micro's TransportRTSP does
 * the same thing; this is the sanctioned escape hatch, not a hack of our own
 * invention. It does couple us to the live555 ABI bundled inside
 * libcvi_rtsp.so (2020.07.21) -- another reason that library is due for
 * replacement.
 *
 * Reaching through that void* also means obeying live555's threading rule: its
 * objects belong to the event loop, and triggerEvent() is the only entry point
 * another thread may use. Both mutations this file makes -- the auth database
 * and each session's ONVIF metadata subsession -- are therefore queued onto the
 * loop and waited for; see LoopExecutor below.
 */

#include "rtsp_server.h"

#include <pthread.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>

#include "app_ipcam_comm.h"
#include "app_ipcam_venc.h"
#include "rtsp.h"
#include <video.h>

#include <liveMedia.hh>

#define RS_TAG "rtsp_server"

namespace {

/* Every live555 object inside CVI_RTSP_CTX belongs to the event-loop thread
 * that CVI_RTSP_Start() spawns; only triggerEvent() may be called from
 * anywhere else. So the two mutations this file needs -- installing the auth
 * database and creating a session plus its metadata subsession -- are packaged
 * as requests, handed to the loop through an event trigger, and waited for.
 *
 * Verified against the shipped SDK (libcvi_rtsp.so, riscv64, disassembled):
 *   - CVI_RTSP_Create() fills ctx->server, ctx->env and ctx->scheduler before
 *     returning, so the trigger can be created while the loop is still stopped
 *     (createEventTrigger() is not itself thread-safe).
 *   - CVI_RTSP_Start() pthread_create()s a thread whose body is
 *     scheduler->doEventLoop(&ctx->stop).
 *   - CVI_RTSP_Stop() sets ctx->stop and pthread_join()s that thread, so after
 *     it returns the trigger can be deleted with no callback in flight.
 *   - The scheduler is built with a 10 ms granularity, which is what bounds
 *     how long select() can sit on a triggerEvent() from another thread. */
struct LoopRequest {
    enum Op { InstallAuth, CreateSessionAndAttach };

    Op op = InstallAuth;
    int idx = -1;

    std::mutex mutex;
    std::condition_variable done;
    bool completed = false;
    bool cancelled = false;
    int result = CVI_FAILURE;
};

struct LoopExecutor {
    TaskScheduler* scheduler = nullptr;
    EventTriggerId trigger_id = 0;

    /* Its own lock, never nested with g_rs.mutex: the handler pops a request
     * and releases this before running it. */
    std::mutex mutex;
    std::deque<std::shared_ptr<LoopRequest>> queue;
    bool accepting = false;
};

struct RtspServerState {
    CVI_RTSP_CTX* ctx = nullptr;
    CVI_RTSP_SESSION* session[RTSP_SERVER_MAX_SESSIONS] = { nullptr };
    CVI_RTSP_SESSION_ATTR attr[RTSP_SERVER_MAX_SESSIONS] = {};
    CVI_RTSP_STATE_LISTENER listener = {};
    VENC_CHN venc_chn[RTSP_SERVER_MAX_SESSIONS] = { 0 };
    int width[RTSP_SERVER_MAX_SESSIONS] = { 0 };
    int height[RTSP_SERVER_MAX_SESSIONS] = { 0 };
    int frame_rate[RTSP_SERVER_MAX_SESSIONS] = { 0 };
    int encoder_bitrate[RTSP_SERVER_MAX_SESSIONS] = { 0 };
    bool started[RTSP_SERVER_MAX_SESSIONS] = { false };
    int session_cnt = 0;

    int port = 8554;
    std::string session_prefix = "live";
    unsigned int bitrate = 30720;
    std::string username;
    std::string password;
    bool metadata_enabled = false;

    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    bool mutex_ready = false;

    LoopExecutor loop;
};

RtspServerState g_rs;

/* Producer/consumer boundary for the metadata RTP track. Inference only ever
 * replaces this latest-value slot; live555 reads it from its own event-loop
 * thread. This deliberately avoids calling live555 from the inference thread,
 * which its scheduler does not support. */
struct MetadataFrameState {
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    std::string xml;
    uint64_t sequence = 0;
};

MetadataFrameState g_metadata;

/* Single source of truth for the metadata frame ceiling. It bounds three
 * things that used to disagree: what rtsp_server_write_metadata() accepts,
 * what the source advertises through maxFrameSize(), and how large live555
 * sizes its OutPacketBuffer. live555's default OutPacketBuffer::maxSize is
 * 60000, so a source advertising more than that had its frames silently
 * truncated into non-well-formed XML. Kept at 128 KiB rather than the 1 MiB
 * the source used to advertise because the buffer is allocated per RTP sink
 * and this value is already two orders of magnitude above a realistic ONVIF
 * analytics frame. */
constexpr unsigned kMetadataMaxFrameSize = 128u * 1024u;

class OnvifMetadataSource final : public FramedSource {
public:
    static OnvifMetadataSource* createNew(UsageEnvironment& env)
    {
        return new OnvifMetadataSource(env);
    }

    unsigned maxFrameSize() const override { return kMetadataMaxFrameSize; }

protected:
    explicit OnvifMetadataSource(UsageEnvironment& env) : FramedSource(env)
    {
        pthread_mutex_lock(&g_metadata.mutex);
        last_sequence_ = g_metadata.sequence;
        pthread_mutex_unlock(&g_metadata.mutex);
    }

    ~OnvifMetadataSource() override
    {
        if (poll_task_ != nullptr) {
            envir().taskScheduler().unscheduleDelayedTask(poll_task_);
        }
    }

    void doGetNextFrame() override
    {
        std::string xml;
        uint64_t sequence = 0;
        pthread_mutex_lock(&g_metadata.mutex);
        sequence = g_metadata.sequence;
        if (sequence != last_sequence_) {
            xml = g_metadata.xml;
        }
        pthread_mutex_unlock(&g_metadata.mutex);

        if (sequence == last_sequence_ || xml.empty()) {
            poll_task_ = envir().taskScheduler().scheduleDelayedTask(
                10000, poll, this);  // 10 ms; no producer-thread live555 call
            return;
        }

        /* Drop rather than truncate: half an XML document is not parseable by
         * any receiver, so a partial frame is worse than a missing one. */
        if (xml.size() > fMaxSize) {
            APP_PROF_LOG_PRINT(LEVEL_ERROR,
                "rtsp: dropping %u-byte ONVIF metadata frame, sink buffer is %u bytes\n",
                static_cast<unsigned>(xml.size()), fMaxSize);
            last_sequence_ = sequence;
            poll_task_ = envir().taskScheduler().scheduleDelayedTask(
                10000, poll, this);
            return;
        }

        last_sequence_ = sequence;
        fFrameSize = static_cast<unsigned>(xml.size());
        fNumTruncatedBytes = 0;
        memcpy(fTo, xml.data(), fFrameSize);
        gettimeofday(&fPresentationTime, nullptr);
        if (fPresentationTime.tv_sec < last_presentation_.tv_sec ||
            (fPresentationTime.tv_sec == last_presentation_.tv_sec &&
             fPresentationTime.tv_usec <= last_presentation_.tv_usec)) {
            fPresentationTime = last_presentation_;
            if (++fPresentationTime.tv_usec >= 1000000) {
                ++fPresentationTime.tv_sec;
                fPresentationTime.tv_usec = 0;
            }
        }
        last_presentation_ = fPresentationTime;
        fDurationInMicroseconds = 0;
        FramedSource::afterGetting(this);
    }

    /* Without this the 10 ms poll keeps firing after the sink has stopped
     * reading and calls afterGetting() on a source that is no longer
     * delivering, which live555 treats as a fatal internal error. */
    void doStopGettingFrames() override
    {
        if (poll_task_ != nullptr) {
            envir().taskScheduler().unscheduleDelayedTask(poll_task_);
            poll_task_ = nullptr;
        }
        FramedSource::doStopGettingFrames();
    }

private:
    static void poll(void* opaque)
    {
        OnvifMetadataSource* source = static_cast<OnvifMetadataSource*>(opaque);
        source->poll_task_ = nullptr;
        /* Guards the race where the task was already dequeued for execution
         * when doStopGettingFrames() ran, so unscheduling could not cancel it. */
        if (!source->isCurrentlyAwaitingData()) return;
        source->doGetNextFrame();
    }

    uint64_t last_sequence_ = 0;
    TaskToken poll_task_ = nullptr;
    struct timeval last_presentation_ = {};
};

class OnvifMetadataSubsession final : public OnDemandServerMediaSubsession {
public:
    static OnvifMetadataSubsession* createNew(UsageEnvironment& env)
    {
        return new OnvifMetadataSubsession(env);
    }

protected:
    explicit OnvifMetadataSubsession(UsageEnvironment& env)
        : OnDemandServerMediaSubsession(env, True) {}

    FramedSource* createNewStreamSource(unsigned, unsigned& est_bitrate) override
    {
        est_bitrate = 128;
        return OnvifMetadataSource::createNew(envir());
    }

    RTPSink* createNewRTPSink(Groupsock* groupsock,
        unsigned char payload_type, FramedSource*) override
    {
        return SimpleRTPSink::createNew(envir(), groupsock, payload_type,
            90000, "application", "vnd.onvif.metadata", 1,
            False, True);
    }
};

/* Event-loop thread only: reached from the trigger handler, in the same
 * callback that created the session, so no DESCRIBE can be answered between
 * the two. */
bool attach_metadata_track(int idx)
{
    if (g_rs.ctx == nullptr || g_rs.ctx->env == nullptr || idx < 0 ||
        idx >= g_rs.session_cnt) {
        return false;
    }
    UsageEnvironment* env = static_cast<UsageEnvironment*>(g_rs.ctx->env);
    RTSPServer* server = static_cast<RTSPServer*>(g_rs.ctx->server);
    ServerMediaSession* media_session = server != nullptr
        ? server->lookupServerMediaSession(g_rs.attr[idx].name)
        : nullptr;
    if (media_session == nullptr) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR,
            "rtsp: live555 session '%s' not found for metadata\n",
            g_rs.attr[idx].name);
        return false;
    }
    /* Global to live555 and read when each RTP sink allocates its buffer, so
     * it has to be raised before the subsession exists. Only ever raised: a
     * larger value already configured elsewhere stays. */
    if (OutPacketBuffer::maxSize < kMetadataMaxFrameSize) {
        OutPacketBuffer::maxSize = kMetadataMaxFrameSize;
    }
    OnvifMetadataSubsession* metadata = OnvifMetadataSubsession::createNew(*env);
    if (metadata == nullptr || !media_session->addSubsession(metadata)) {
        if (metadata != nullptr) Medium::close(metadata);
        APP_PROF_LOG_PRINT(LEVEL_ERROR,
            "rtsp: failed to attach ONVIF metadata to '%s'\n",
            g_rs.attr[idx].name);
        return false;
    }
    APP_PROF_LOG_PRINT(LEVEL_INFO,
        "rtsp: ONVIF metadata track attached to '%s'\n",
        g_rs.attr[idx].name);
    return true;
}

/* PAYLOAD_TYPE_E -> CVI_RTSP_VIDEO_CODEC. Returns false on an unsupported
 * type (the old APP_RTSP_VCODEC_CHK macro returned CVI_FAILURE from inside the
 * caller, which made it impossible to keep going with the other sessions). */
bool codec_of(PAYLOAD_TYPE_E in, CVI_RTSP_VIDEO_CODEC* out)
{
    switch (in) {
    case PT_H265:
        *out = RTSP_VIDEO_H265;
        return true;
    case PT_H264:
        *out = RTSP_VIDEO_H264;
        return true;
    case PT_MJPEG:
        *out = RTSP_VIDEO_JPEG;
        return true;
    default:
        *out = RTSP_VIDEO_NONE;
        return false;
    }
}

void on_connect(const char* ip, CVI_VOID* arg)
{
    (void)arg;
    APP_PROF_LOG_PRINT(LEVEL_INFO, "rtsp client connected: %s\n", ip);

    /* cvi_rtsp reuses the live VENC source, so a client can join in the middle
     * of a GOP without receiving the SPS/PPS needed to decode it. Ask every
     * published encoder channel for a fresh IDR; the VENC emits its parameter
     * sets with that access unit and all connected clients can resynchronise.
     * requestVideoIDR() is non-blocking/coalesced and is already used by the
     * debug stream for the same late-join case.
     *
     * This callback runs on the live555 event-loop thread while session_cnt
     * and venc_chn[] are written by the application thread, so take a copy
     * under g_rs.mutex and drop the lock before calling into the VENC. Holding
     * it across requestVideoIDR() would put a foreign subsystem's locks
     * underneath ours on this path only. */
    int cnt = 0;
    VENC_CHN chn[RTSP_SERVER_MAX_SESSIONS] = { 0 };
    if (g_rs.mutex_ready) {
        pthread_mutex_lock(&g_rs.mutex);
    }
    cnt = std::min(g_rs.session_cnt, RTSP_SERVER_MAX_SESSIONS);
    for (int i = 0; i < cnt; ++i) {
        chn[i] = g_rs.venc_chn[i];
    }
    if (g_rs.mutex_ready) {
        pthread_mutex_unlock(&g_rs.mutex);
    }

    for (int i = 0; i < cnt; ++i) {
        requestVideoIDR(static_cast<video_ch_index_t>(chn[i]));
    }
}

void on_disconnect(const char* ip, CVI_VOID* arg)
{
    (void)arg;
    APP_PROF_LOG_PRINT(LEVEL_INFO, "rtsp client disconnected: %s\n", ip);
}

/* Event-loop thread only. Install a live555 auth database on the server hidden
 * inside CVI_RTSP_CTX. Runs before any session is published, so there is no
 * window in which a stream exists without its credentials. */
int install_auth_on_loop(void)
{
    if (g_rs.username.empty() || g_rs.password.empty()) {
        return CVI_SUCCESS;  // no credentials configured is not a failure
    }
    if (g_rs.ctx == nullptr) {
        return CVI_FAILURE;
    }
    RTSPServer* server = static_cast<RTSPServer*>(g_rs.ctx->server);
    if (server == nullptr) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR, "rtsp: no live555 server, auth not installed\n");
        return CVI_FAILURE;
    }
    UserAuthenticationDatabase* auth = new (std::nothrow) UserAuthenticationDatabase();
    if (auth == nullptr) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR, "rtsp: out of memory installing auth\n");
        return CVI_FAILURE;
    }
    auth->addUserRecord(g_rs.username.c_str(), g_rs.password.c_str());
    server->setAuthenticationDatabase(auth);
    APP_PROF_LOG_PRINT(LEVEL_INFO, "rtsp: authentication enabled for user '%s'\n",
        g_rs.username.c_str());
    return CVI_SUCCESS;
}

/* Event-loop thread only. Creating the session and attaching its metadata
 * subsession happen in one callback; only once both succeeded is the session
 * published to the other threads, so a DESCRIBE either misses the stream
 * entirely or sees it complete. */
int create_session_on_loop(int idx)
{
    if (g_rs.ctx == nullptr || idx < 0 || idx >= g_rs.session_cnt) {
        return CVI_FAILURE;
    }
    CVI_RTSP_SESSION* session = nullptr;
    CVI_S32 ret = CVI_RTSP_CreateSession(g_rs.ctx, &g_rs.attr[idx], &session);
    if (ret < 0 || session == nullptr) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR,
            "rtsp: CVI_RTSP_CreateSession('%s') failed: %d\n",
            g_rs.attr[idx].name, ret);
        return CVI_FAILURE;
    }
    if (g_rs.metadata_enabled && !attach_metadata_track(idx)) {
        CVI_RTSP_DestroySession(g_rs.ctx, session);
        return CVI_FAILURE;
    }

    /* The only place this file takes g_rs.mutex from the event loop besides
     * on_connect(), and it is held for two stores. */
    if (g_rs.mutex_ready) {
        pthread_mutex_lock(&g_rs.mutex);
    }
    g_rs.session[idx] = session;
    g_rs.started[idx] = true;
    if (g_rs.mutex_ready) {
        pthread_mutex_unlock(&g_rs.mutex);
    }
    APP_PROF_LOG_PRINT(LEVEL_INFO, "rtsp: session '%s' published\n",
        g_rs.attr[idx].name);
    return CVI_SUCCESS;
}

const char* op_name(LoopRequest::Op op)
{
    return op == LoopRequest::InstallAuth ? "InstallAuth" : "CreateSessionAndAttach";
}

unsigned long long this_thread_id(void)
{
    return static_cast<unsigned long long>(
        reinterpret_cast<uintptr_t>(reinterpret_cast<void*>(pthread_self())));
}

/* live555 may coalesce several triggerEvent() calls into one handler run, so
 * drain the whole queue every time. */
void loop_handler(void* client_data)
{
    (void)client_data;
    for (;;) {
        std::shared_ptr<LoopRequest> req;
        {
            std::lock_guard<std::mutex> guard(g_rs.loop.mutex);
            if (g_rs.loop.queue.empty()) {
                return;
            }
            req = g_rs.loop.queue.front();
            g_rs.loop.queue.pop_front();
        }

        bool cancelled = false;
        {
            std::lock_guard<std::mutex> guard(req->mutex);
            cancelled = req->cancelled;
        }
        if (cancelled) {
            APP_PROF_LOG_PRINT(LEVEL_WARN,
                "rtsp: loop request %s idx=%d was cancelled, skipping\n",
                op_name(req->op), req->idx);
            continue;
        }

        APP_PROF_LOG_PRINT(LEVEL_INFO,
            "rtsp: loop request %s idx=%d begin on thread %llu\n",
            op_name(req->op), req->idx, this_thread_id());
        int result = CVI_FAILURE;
        switch (req->op) {
        case LoopRequest::InstallAuth:
            result = install_auth_on_loop();
            break;
        case LoopRequest::CreateSessionAndAttach:
            result = create_session_on_loop(req->idx);
            break;
        }
        APP_PROF_LOG_PRINT(LEVEL_INFO,
            "rtsp: loop request %s idx=%d end, result=%d\n",
            op_name(req->op), req->idx, result);

        {
            std::lock_guard<std::mutex> guard(req->mutex);
            req->result = result;
            req->completed = true;
        }
        req->done.notify_all();
    }
}

/* Called on the application thread while the loop is still stopped:
 * createEventTrigger() mutates scheduler state that only the loop may touch
 * once it is running. */
bool loop_executor_init(void)
{
    if (g_rs.ctx == nullptr) {
        return false;
    }
    UsageEnvironment* env = static_cast<UsageEnvironment*>(g_rs.ctx->env);
    if (env == nullptr) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR,
            "rtsp: CVI_RTSP_Create() left no live555 environment\n");
        return false;
    }
    TaskScheduler* scheduler = &env->taskScheduler();
    EventTriggerId trigger_id = scheduler->createEventTrigger(loop_handler);
    if (trigger_id == 0) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR, "rtsp: no live555 event trigger available\n");
        return false;
    }
    std::lock_guard<std::mutex> guard(g_rs.loop.mutex);
    g_rs.loop.scheduler = scheduler;
    g_rs.loop.trigger_id = trigger_id;
    g_rs.loop.queue.clear();
    g_rs.loop.accepting = true;
    APP_PROF_LOG_PRINT(LEVEL_INFO,
        "rtsp: live555 event trigger %u created before the loop started\n",
        static_cast<unsigned>(trigger_id));
    return true;
}

/* Queue one mutation for the event loop and block until it reports back.
 * Deliberately holds neither g_rs.mutex nor the queue lock while waiting: the
 * loop thread takes g_rs.mutex in on_connect() and in create_session_on_loop(),
 * so waiting under it would be an ABBA deadlock. */
int run_on_loop_and_wait(LoopRequest::Op op, int idx)
{
    TaskScheduler* scheduler = nullptr;
    EventTriggerId trigger_id = 0;
    auto req = std::make_shared<LoopRequest>();
    req->op = op;
    req->idx = idx;
    {
        std::lock_guard<std::mutex> guard(g_rs.loop.mutex);
        if (!g_rs.loop.accepting || g_rs.loop.scheduler == nullptr ||
            g_rs.loop.trigger_id == 0) {
            APP_PROF_LOG_PRINT(LEVEL_ERROR,
                "rtsp: loop request %s idx=%d rejected, executor not running\n",
                op_name(op), idx);
            return CVI_FAILURE;
        }
        scheduler = g_rs.loop.scheduler;
        trigger_id = g_rs.loop.trigger_id;
        g_rs.loop.queue.push_back(req);
    }
    APP_PROF_LOG_PRINT(LEVEL_INFO,
        "rtsp: loop request %s idx=%d queued from thread %llu\n",
        op_name(op), idx, this_thread_id());

    const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    scheduler->triggerEvent(trigger_id, nullptr);

    std::unique_lock<std::mutex> lock(req->mutex);
    const bool signalled = req->done.wait_for(lock, std::chrono::seconds(3),
        [&req] { return req->completed; });
    const long long elapsed_ms = static_cast<long long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - begin).count());
    if (!signalled) {
        /* Leave the request in the queue: the handler drops it on sight, and
         * the shared_ptr keeps it alive until then. */
        req->cancelled = true;
        lock.unlock();
        APP_PROF_LOG_PRINT(LEVEL_ERROR,
            "rtsp: loop request %s idx=%d timed out after %lld ms\n",
            op_name(op), idx, elapsed_ms);
        return CVI_FAILURE;
    }
    const int result = req->result;
    lock.unlock();
    APP_PROF_LOG_PRINT(LEVEL_INFO,
        "rtsp: loop request %s idx=%d done in %lld ms, result=%d\n",
        op_name(op), idx, elapsed_ms, result);
    return result;
}

/* Stop accepting work and release anyone still waiting, before the loop is
 * asked to exit -- otherwise their 3 s timeout is the only way out. */
void loop_executor_drain(void)
{
    std::deque<std::shared_ptr<LoopRequest>> pending;
    {
        std::lock_guard<std::mutex> guard(g_rs.loop.mutex);
        g_rs.loop.accepting = false;
        pending.swap(g_rs.loop.queue);
    }
    for (const std::shared_ptr<LoopRequest>& req : pending) {
        {
            std::lock_guard<std::mutex> guard(req->mutex);
            req->cancelled = true;
            req->completed = true;
            req->result = CVI_FAILURE;
        }
        req->done.notify_all();
    }
}

/* Only safe once CVI_RTSP_Stop() has joined the loop thread. */
void loop_executor_release(void)
{
    std::lock_guard<std::mutex> guard(g_rs.loop.mutex);
    if (g_rs.loop.scheduler != nullptr && g_rs.loop.trigger_id != 0) {
        g_rs.loop.scheduler->deleteEventTrigger(g_rs.loop.trigger_id);
        APP_PROF_LOG_PRINT(LEVEL_INFO, "rtsp: live555 event trigger %u deleted\n",
            static_cast<unsigned>(g_rs.loop.trigger_id));
    }
    g_rs.loop.scheduler = nullptr;
    g_rs.loop.trigger_id = 0;
    g_rs.loop.queue.clear();
}

} // namespace

extern "C" {

void rtsp_server_config_init(rtsp_server_config_t* cfg)
{
    if (cfg == nullptr) {
        return;
    }
    cfg->port = 8554;
    cfg->session_prefix = "live";
    cfg->bitrate = 30720;
    cfg->username = nullptr;
    cfg->password = nullptr;
    cfg->ch_mask = 0;
    cfg->metadata_enabled = false;
}

int rtsp_server_start(const rtsp_server_config_t* cfg)
{
    if (g_rs.ctx != nullptr) {
        APP_PROF_LOG_PRINT(LEVEL_WARN, "rtsp server has been created\n");
        return CVI_SUCCESS;
    }

    rtsp_server_config_t def;
    rtsp_server_config_init(&def);
    if (cfg == nullptr) {
        cfg = &def;
    }

    g_rs.port = cfg->port > 0 ? cfg->port : def.port;
    g_rs.session_prefix = (cfg->session_prefix && cfg->session_prefix[0])
        ? cfg->session_prefix
        : def.session_prefix;
    g_rs.bitrate = cfg->bitrate > 0 ? cfg->bitrate : def.bitrate;
    g_rs.username = cfg->username ? cfg->username : "";
    g_rs.password = cfg->password ? cfg->password : "";
    g_rs.metadata_enabled = cfg->metadata_enabled;

    /* Channel selection: bit i -> channel i, packed in ascending order. */
    g_rs.session_cnt = 0;
    for (int i = 0; i < RTSP_SERVER_MAX_SESSIONS; i++) {
        if (cfg->ch_mask & (0x1 << i)) {
            g_rs.venc_chn[g_rs.session_cnt] = i;
            g_rs.session_cnt++;
        }
    }
    if (g_rs.session_cnt == 0) {
        APP_PROF_LOG_PRINT(LEVEL_WARN, "rtsp: no channel selected (ch_mask=0)\n");
        return CVI_SUCCESS;
    }

    /* Per-session attributes: codec comes from the channel's VENC config. */
    APP_PARAM_VENC_CTX_S* venc = app_ipcam_Venc_Param_Get();
    for (int i = 0; i < g_rs.session_cnt; i++) {
        memset(&g_rs.attr[i], 0, sizeof(g_rs.attr[i]));
        g_rs.attr[i].video.bitrate = g_rs.bitrate;

        APP_VENC_CHN_CFG_S* chn_cfg = &venc->astVencChnCfg[g_rs.venc_chn[i]];
        g_rs.width[i] = static_cast<int>(chn_cfg->u32Width);
        g_rs.height[i] = static_cast<int>(chn_cfg->u32Height);
        g_rs.frame_rate[i] = static_cast<int>(chn_cfg->u32DstFrameRate);
        g_rs.encoder_bitrate[i] = static_cast<int>(chn_cfg->u32BitRate);
        if (!codec_of(chn_cfg->enType, &g_rs.attr[i].video.codec)) {
            APP_PROF_LOG_PRINT(LEVEL_ERROR,
                "rtsp: VencChn_%d payload type %d unsupported\n",
                g_rs.venc_chn[i], chn_cfg->enType);
            return CVI_FAILURE;
        }
#ifdef AUDIO_SUPPORT
        APP_PARAM_AUDIO_CFG_T* audio_cfg = app_ipcam_Audio_Param_Get();
        if (audio_cfg->bInit) {
            g_rs.attr[i].audio.codec = RTSP_AUDIO_PCM_L16;
            g_rs.attr[i].audio.sampleRate = audio_cfg->astAudioCfg.enSamplerate;
        }
#endif
        APP_PROF_LOG_PRINT(LEVEL_INFO,
            "VencChn_%d attach to Session_%d with CodecType=%d\n",
            g_rs.venc_chn[i], i, g_rs.attr[i].video.codec);
    }

    CVI_RTSP_CONFIG config = {};
    config.port = g_rs.port;

    CVI_S32 ret = CVI_RTSP_Create(&g_rs.ctx, &config);
    if (ret < 0) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR, "fail to create rtsp\n");
        return ret;
    }

    APP_PROF_LOG_PRINT(LEVEL_INFO,
        "rtsp: cvi_rtsp context created (server=%p env=%p scheduler=%p)\n",
        g_rs.ctx->server, g_rs.ctx->env, g_rs.ctx->scheduler);

    pthread_mutex_init(&g_rs.mutex, NULL);
    g_rs.mutex_ready = true;

    /* Everything that touches live555 from here on goes through the loop
     * executor, so its trigger has to exist before the loop does. */
    if (!loop_executor_init()) {
        rtsp_server_stop();
        return CVI_FAILURE;
    }

    /* Installed before Start() rather than after: it is one more write to the
     * context, and doing it now keeps it off the running loop. */
    g_rs.listener.onConnect = on_connect;
    g_rs.listener.argConn = g_rs.ctx;
    g_rs.listener.onDisconnect = on_disconnect;
    CVI_RTSP_SetListener(g_rs.ctx, &g_rs.listener);

    /* Session names must be settled before the loop can be asked to create
     * them: the request carries only the index. */
    for (int i = 0; i < g_rs.session_cnt; i++) {
        snprintf(g_rs.attr[i].name, sizeof(g_rs.attr[i].name), "%s%d",
            g_rs.session_prefix.c_str(), i);
        g_rs.attr[i].reuseFirstSource = 1;
    }

    ret = CVI_RTSP_Start(g_rs.ctx);
    if (ret < 0) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR, "fail to rtsp start\n");
        rtsp_server_stop();
        return ret;
    }
    APP_PROF_LOG_PRINT(LEVEL_INFO, "rtsp: event loop started on port %d\n", g_rs.port);

    /* Credentials first: a session published before the auth database exists
     * would be reachable without one. */
    if (run_on_loop_and_wait(LoopRequest::InstallAuth, -1) != CVI_SUCCESS) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR, "rtsp: authentication could not be installed\n");
        rtsp_server_stop();
        return CVI_FAILURE;
    }

    for (int i = 0; i < g_rs.session_cnt; i++) {
        if (run_on_loop_and_wait(LoopRequest::CreateSessionAndAttach, i) != CVI_SUCCESS) {
            APP_PROF_LOG_PRINT(LEVEL_ERROR,
                "rtsp: session '%s' could not be created\n", g_rs.attr[i].name);
            rtsp_server_stop();
            return CVI_FAILURE;
        }
        APP_PROF_LOG_PRINT(LEVEL_INFO, "======rtsp start [VencChn%d  %s]  ======\n",
            g_rs.venc_chn[i], g_rs.attr[i].name);
    }

    return CVI_SUCCESS;
}

void rtsp_server_stop(void)
{
    if (g_rs.ctx == nullptr) {
        printf("rtsp server has not been create\n");
        return;
    }

    /* Refuse new work and release whoever is already queued, so nobody is left
     * waiting on a loop that is about to exit. */
    loop_executor_drain();

    APP_PROF_LOG_PRINT(LEVEL_INFO, "rtsp: stopping event loop\n");
    CVI_RTSP_Stop(g_rs.ctx);
    APP_PROF_LOG_PRINT(LEVEL_INFO, "rtsp: event loop thread joined\n");

    /* CVI_RTSP_Stop() joins the loop thread, so no callback can be in flight
     * and the trigger can go before CVI_RTSP_Destroy() frees the scheduler. */
    loop_executor_release();

    if (g_rs.mutex_ready) {
        pthread_mutex_lock(&g_rs.mutex);
    }
    for (int i = 0; i < g_rs.session_cnt; i++) {
        if (g_rs.started[i]) {
            CVI_RTSP_DestroySession(g_rs.ctx, g_rs.session[i]);
            g_rs.started[i] = false;
            g_rs.session[i] = nullptr;
        }
    }
    if (g_rs.mutex_ready) {
        pthread_mutex_unlock(&g_rs.mutex);
    }

    CVI_RTSP_Destroy(&g_rs.ctx);
    APP_PROF_LOG_PRINT(LEVEL_INFO, "rtsp server destroyed\n");

    if (g_rs.mutex_ready) {
        /* Clear the fields other threads read under this lock while the lock
         * still exists, then retire it. */
        pthread_mutex_lock(&g_rs.mutex);
        g_rs.ctx = nullptr;
        g_rs.session_cnt = 0;
        g_rs.metadata_enabled = false;
        pthread_mutex_unlock(&g_rs.mutex);
        pthread_mutex_destroy(&g_rs.mutex);
        g_rs.mutex_ready = false;
    } else {
        g_rs.ctx = nullptr;
        g_rs.session_cnt = 0;
        g_rs.metadata_enabled = false;
    }
    pthread_mutex_lock(&g_metadata.mutex);
    g_metadata.xml.clear();
    ++g_metadata.sequence;
    pthread_mutex_unlock(&g_metadata.mutex);
}

int rtsp_server_video_handler(void* pData, void* pArgs, void* pUserData)
{
    (void)pUserData;

    APP_DATA_CTX_S* data_ctx = (APP_DATA_CTX_S*)pArgs;
    APP_DATA_PARAM_S* data_param = &data_ctx->stDataParam;
    APP_VENC_CHN_CFG_S* chn_cfg = (APP_VENC_CHN_CFG_S*)data_param->pParam;
    VENC_CHN chn = chn_cfg->VencChn;

    int idx = 0;
    for (int i = 0; i < g_rs.session_cnt; i++) {
        if (g_rs.venc_chn[i] == chn) {
            idx = i;
            break;
        }
    }
    if (!g_rs.started[idx]) {
        return CVI_SUCCESS;
    }

    VENC_STREAM_S* stream = (VENC_STREAM_S*)pData;
    if (stream->u32PackCount == 0) {
        APP_PROF_LOG_PRINT(LEVEL_ERROR, "pstStream->u32PackCount is %d\n",
            stream->u32PackCount);
        return CVI_SUCCESS;
    }

    CVI_RTSP_DATA data;
    memset(&data, 0, sizeof(data));
    data.blockCnt = stream->u32PackCount;
    for (CVI_U32 i = 0; i < stream->u32PackCount; i++) {
        VENC_PACK_S* pack = &stream->pstPack[i];
        data.dataPtr[i] = pack->pu8Addr + pack->u32Offset;
        data.dataLen[i] = pack->u32Len - pack->u32Offset;
    }

    if (g_rs.ctx != nullptr && g_rs.session[idx] != nullptr) {
        if (CVI_RTSP_WriteFrame(g_rs.ctx, g_rs.session[idx]->video, &data) != CVI_SUCCESS) {
            APP_PROF_LOG_PRINT(LEVEL_ERROR, "CVI_RTSP_WriteFrame failed\n");
        }
    }

    return CVI_SUCCESS;
}

int rtsp_server_port(void) { return g_rs.port; }

int rtsp_server_session_count(void) { return g_rs.session_cnt; }

const char* rtsp_server_session_name(int idx)
{
    if (idx < 0 || idx >= g_rs.session_cnt) {
        return NULL;
    }
    return g_rs.attr[idx].name;
}

int rtsp_server_width(int idx)
{
    return idx >= 0 && idx < g_rs.session_cnt ? g_rs.width[idx] : 0;
}

int rtsp_server_height(int idx)
{
    return idx >= 0 && idx < g_rs.session_cnt ? g_rs.height[idx] : 0;
}

int rtsp_server_frame_rate(int idx)
{
    return idx >= 0 && idx < g_rs.session_cnt ? g_rs.frame_rate[idx] : 0;
}

int rtsp_server_encoder_bitrate(int idx)
{
    return idx >= 0 && idx < g_rs.session_cnt ? g_rs.encoder_bitrate[idx] : 0;
}

bool rtsp_server_auth_enabled(void)
{
    return !g_rs.username.empty() && !g_rs.password.empty();
}

bool rtsp_server_metadata_enabled(void)
{
    /* Called from the inference thread; ctx and metadata_enabled are cleared
     * by rtsp_server_stop() on the application thread under g_rs.mutex. */
    if (!g_rs.mutex_ready) {
        return false;
    }
    pthread_mutex_lock(&g_rs.mutex);
    const bool enabled = g_rs.ctx != nullptr && g_rs.metadata_enabled;
    pthread_mutex_unlock(&g_rs.mutex);
    return enabled;
}

int rtsp_server_write_metadata(const char* xml, size_t len)
{
    if (!rtsp_server_metadata_enabled() || xml == nullptr || len == 0 ||
        len > kMetadataMaxFrameSize) {
        return -1;
    }
    pthread_mutex_lock(&g_metadata.mutex);
    g_metadata.xml.assign(xml, len);
    ++g_metadata.sequence;
    pthread_mutex_unlock(&g_metadata.mutex);
    return 0;
}

int rtsp_server_url(char* buf, size_t buflen, const char* host, int idx)
{
    const char* name = rtsp_server_session_name(idx);
    if (buf == NULL || buflen == 0 || host == NULL || name == NULL) {
        return -1;
    }
    int n = rtsp_server_auth_enabled()
        ? snprintf(buf, buflen, "rtsp://%s@%s:%d/%s", g_rs.username.c_str(), host, g_rs.port, name)
        : snprintf(buf, buflen, "rtsp://%s:%d/%s", host, g_rs.port, name);
    return (n < 0 || (size_t)n >= buflen) ? -1 : n;
}

/* ---- back-compat shims ------------------------------------------------- */

int initRtsp(uint8_t chEnableFlag)
{
    rtsp_server_config_t cfg;
    rtsp_server_config_init(&cfg);
    cfg.ch_mask = chEnableFlag;
    rtsp_server_start(&cfg);
    return 0;
}

int deinitRtsp(void)
{
    rtsp_server_stop();
    return 0;
}

int fpStreamingSendToRtsp(void* pData, void* pArgs, void* pUserData)
{
    return rtsp_server_video_handler(pData, pArgs, pUserData);
}

} // extern "C"
