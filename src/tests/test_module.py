import pytest
from flask_app.app import app as flask_app, socketio

@pytest.fixture
def app():
    yield flask_app

@pytest.fixture
def client(app):
    return socketio.test_client(app)

# @pytest.fixture
# def dynamic_event_data():
#     return {
#         'landmarkHistory': 
#             tensor([[-4.5559, -2.7756, -7.0466, -6.9591, -6.9683, -7.7638, -6.9739, -7.1603,
#                     -6.6906, -0.0945, -6.9080, -6.9301, -7.6667, -8.1346, -7.9324, -7.4257,
#                     -7.6182, -7.0100, -8.3081, -6.3362, -8.1397, -7.0010, -7.1553, -6.7354,
#                     -7.2159, -8.5038]], grad_fn=<LogSoftmaxBackward0>)
#             9 0.936595800769773
#         }

@pytest.fixture
def stream_event_data():
    return {'features':
            [[0.0, 0.0], [-0.05054670572280884, -0.005826413631439209], [-0.11023426055908203, 0.05315476655960083],
             [-0.11378547549247742, 0.13173598051071167], [-0.08076173067092896, 0.17083382606506348],
             [-0.09057804942131042, 0.14104223251342773], [-0.09843555092811584, 0.1867084801197052],
             [-0.0873461365699768, 0.12149304151535034], [-0.08362278342247009, 0.09948885440826416],
             [-0.05295476317405701, 0.15066349506378174], [-0.05809706449508667, 0.18631955981254578],
             [-0.05212736129760742, 0.107901930809021], [-0.05124577879905701, 0.10319015383720398],
             [-0.016105592250823975, 0.14986467361450195], [-0.01995319128036499, 0.18234413862228394],
             [-0.019071906805038452, 0.11091336607933044], [-0.01757708191871643, 0.10414722561836243],
             [0.023833155632019043, 0.14780771732330322], [0.016992241144180298, 0.16973933577537537],
             [0.014041721820831299, 0.11977922916412354], [0.016810595989227295, 0.10923758149147034]]
        }

# def test_dynamic_event(client, dynamic_event_data):
#     client.emit('dynamic', dynamic_event_data)
#     received = client.get_received()
#     assert received[0]['name'] == 'dynamic'
#     assert 'result' in received[0]['args'][0]

def test_stream_event(client, stream_event_data):
    client.emit('stream', stream_event_data)
    received = client.get_received()
    assert received[0]['name'] == 'stream'
    assert 'result' in received[0]['args'][0]
    expected_result = 'A'
    assert received[0]['args'][0]['result'] == expected_result