#include "onvif_meta.h"

#include <cassert>
#include <iostream>
#include <string>

namespace {

void contains(const std::string& value, const std::string& expected)
{
    assert(value.find(expected) != std::string::npos);
}

void test_empty_frame()
{
    onvif_frame_t frame;
    frame.utc_ms = 0;

    const std::string xml = onvif_meta_to_xml(frame);
    contains(xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
    contains(xml, "<tt:MetadataStream");
    contains(xml, "xmlns:tt=\"http://www.onvif.org/ver10/schema\"");
    contains(xml, "xmlns:fc=\"http://www.onvif.org/ver20/analytics/humanface\"");
    contains(xml, "xmlns:recam=\"http://www.seeedstudio.com/recamera/schema\"");
    contains(xml, "<tt:VideoAnalytics><tt:Frame UtcTime=\"1970-01-01T00:00:00.000Z\">");
    contains(xml, "</tt:Frame></tt:VideoAnalytics></tt:MetadataStream>");
    assert(xml.find("<tt:Object") == std::string::npos);
    assert(xml.find("<tt:Transformation>") == std::string::npos);
}

void test_human_box_and_transform()
{
    onvif_frame_t frame;
    frame.utc_ms = 1786240923321ull;
    frame.source = "YoloDetector";
    frame.frame_w = 1920;
    frame.frame_h = 1080;

    onvif_object_t human;
    human.id = 12;
    human.parent = 7;
    human.cx = 700.f;
    human.cy = 460.f;
    human.w = 160.f;
    human.h = 520.f;
    human.classes.push_back({"Human", 0.93f});
    frame.objects.push_back(human);

    const std::string xml = onvif_meta_to_xml(frame);
    contains(xml, " Source=\"YoloDetector\"");
    contains(xml, "<tt:Translate x=\"-1.0\" y=\"-1.0\"/>");
    contains(xml, "<tt:Scale x=\"0.00104167\" y=\"0.00185185\"/>");
    contains(xml, "<tt:Object ObjectId=\"12\" Parent=\"7\">");
    contains(xml, "<tt:BoundingBox left=\"620.00\" top=\"880.00\" right=\"780.00\" bottom=\"360.00\"/>");
    contains(xml, "<tt:CenterOfGravity x=\"700.00\" y=\"620.00\"/>");
    contains(xml, "<tt:Class><tt:Type Likelihood=\"0.930\">Human</tt:Type></tt:Class>");
}

void test_escaping_and_optional_fields()
{
    onvif_frame_t frame;
    frame.utc_ms = 1;
    frame.source = "A&B <camera> \"one\" 'two'";

    onvif_object_t object;
    object.id = 3;
    object.cx = 0.f;
    object.cy = 0.f;
    object.w = 2.f;
    object.h = 2.f;
    object.classes.push_back({"Human & <guest>", 0.5f});
    object.face.present = true;
    object.face.age_min = 25;
    object.face.age_max = 35;
    object.face.gender = "Female & guest";
    object.barcode_data = "https://example.test/?a=1&b=<2>";
    object.barcode_type = "QR&Code";
    object.extensions.push_back({"Emotion", "Neutral & <calm> \"ok\" 'yes'"});
    object.extensions.push_back({"bad:name", "must be skipped"});
    frame.objects.push_back(object);

    const std::string xml = onvif_meta_to_xml(frame);
    contains(xml, "Source=\"A&amp;B &lt;camera&gt; &quot;one&quot; &apos;two&apos;\"");
    contains(xml, ">Human &amp; &lt;guest&gt;</tt:Type>");
    contains(xml, "<tt:HumanFace><fc:Age><tt:Min>25</tt:Min><tt:Max>35</tt:Max></fc:Age><fc:Gender>Female &amp; guest</fc:Gender></tt:HumanFace>");
    contains(xml, "<tt:BarcodeInfo><tt:Data>https://example.test/?a=1&amp;b=&lt;2&gt;</tt:Data><tt:Type>QR&amp;Code</tt:Type></tt:BarcodeInfo>");
    contains(xml, "<recam:Emotion>Neutral &amp; &lt;calm&gt; &quot;ok&quot; &apos;yes&apos;</recam:Emotion>");
    assert(xml.find("bad:name") == std::string::npos);

    const size_t face = xml.find("<tt:HumanFace>");
    const size_t barcode = xml.find("<tt:BarcodeInfo>");
    const size_t extension = xml.find("<recam:Emotion>");
    assert(face < barcode && barcode < extension);
}

} // namespace

int main()
{
    test_empty_frame();
    test_human_box_and_transform();
    test_escaping_and_optional_fields();
    std::cout << "onvif_meta_xml_test: PASS\n";
    return 0;
}
