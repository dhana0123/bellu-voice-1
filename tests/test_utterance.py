from bellu.utterance import merge_partial


def test_merge_partial_grows_and_dedupes():
    assert merge_partial("", "హలో") == "హలో"
    assert merge_partial("హలో", "హలో టెస్టింగ్") == "హలో టెస్టింగ్"
    assert merge_partial("హలో టెస్టింగ్", "టెస్టింగ్") == "హలో టెస్టింగ్"
    assert merge_partial("వన్ టూ", "టూ త్రీ") == "వన్ టూ త్రీ"
