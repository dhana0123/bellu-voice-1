from bellu.utterance import UtteranceTracker, merge_partial


def test_ingest_rejects_duplicate_text():
    t = UtteranceTracker()
    assert t.begin_speech(100, preroll=0)
    assert t.ingest_asr("హలో", 200)
    assert not t.ingest_asr("హలో", 400)
    assert t.current.text == "హలో"


def test_freeze_stops_asr_ingest():
    t = UtteranceTracker()
    t.begin_speech(0, preroll=0)
    t.ingest_asr("అయోధ్య", 10)
    assert t.freeze(20)
    assert not t.open()
    assert not t.ingest_asr("అయోధ్య", 30)


def test_merge_partial_grows_and_dedupes():
    assert merge_partial("", "హలో") == "హలో"
    assert merge_partial("హలో", "హలో టెస్టింగ్") == "హలో టెస్టింగ్"
    assert merge_partial("హలో టెస్టింగ్", "టెస్టింగ్") == "హలో టెస్టింగ్"
    assert merge_partial("వన్ టూ", "టూ త్రీ") == "వన్ టూ త్రీ"
