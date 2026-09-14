from bellu.brain.mock import MockBrain
from bellu.chat import ChatSession


def test_chat_session_replies():
    session = ChatSession(brain=MockBrain())
    assert session.reply("hello") == "I heard you: hello"
    assert session.history[-1]["role"] == "assistant"
