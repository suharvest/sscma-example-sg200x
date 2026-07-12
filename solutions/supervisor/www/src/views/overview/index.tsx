import { useEffect, useRef, useState } from 'react'
import moment from 'moment'
import { useTranslation } from 'react-i18next'
import { getWebSocketUrlApi } from '@/api'
import useDebugStream from '@/hooks/useDebugStream'

function Home() {
	const { t } = useTranslation()
	const videoRef = useRef<HTMLVideoElement>(null)
	const [wsUrl, setWsUrl] = useState<string | null>(null)

	useEffect(() => {
		getWebSocketUrlApi({ time: Date.now() })
			.then(({ data }) => setWsUrl(data.websocketUrl))
			.catch((err) => console.log('err:', err))
	}, [])

	const { lastFrameDelay, lastFrameTs } = useDebugStream({
		enabled: true,
		wsUrl,
		resultsUrl: null,
		videoRef,
	})

	return (
		<div className='m-auto p-16' style={{ maxWidth: '600px' }}>
			<div className='iframe my-20  flex justify-center' style={{ height: 'auto' }}>
				<video
					ref={videoRef}
					className='rounded-20'
					width='100%'
					muted
					autoPlay
					controls
				></video>
			</div>
			<div className='flex justify-between text-black opacity-60 mb-10'>
				<span>{t('overview.timeStamp')}</span>
				<span>{t('overview.delay')}</span>
			</div>

			<div className='flex justify-between text-17 '>
				<span>{moment(lastFrameTs || 0).format('YYYY-MM-DD hh:mm:ss')}</span>
				<span>{lastFrameDelay ?? 0}ms</span>
			</div>
		</div>
	)
}

export default Home
