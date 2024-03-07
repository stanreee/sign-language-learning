import pytest
from app import app as flask_app, socketio

@pytest.fixture
def app():
    yield flask_app

@pytest.fixture
def client(app):
    return socketio.test_client(app)

@pytest.fixture
def dynamic_event_data():
    return {
        'landmarkHistory': 
    }

@pytest.fixture
def stream_event_data():
    return {
        'features': 
    }

def test_dynamic_event(client, dynamic_event_data):
    client.emit('dynamic', dynamic_event_data)
    received = client.get_received()
    assert received[0]['name'] == 'dynamic'
    assert 'result' in received[0]['args'][0]

def test_stream_event(client, stream_event_data):
    client.emit('stream', stream_event_data)
    received = client.get_received()
    assert received[0]['name'] == 'stream'
    assert 'result' in received[0]['args'][0]
    # Optionally, you can also assert the actual result
    # expected_result = 'A'  # or whatever your expected result is
    # assert received[0]['args'][0]['result'] == expected_result