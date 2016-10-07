#encoding=utf-8

import http.server
import json
import numpy as np
from ocr import OCRNeuralNetwork
from neural_network_design import test
from sklearn.cross_validation import train_test_split

HOST_NAME         = 'localhost'
PORT              = 8000
HIDDEN_NODE_COUNT = 25

# Load Data
print("Loading Data ... ")
data_matrix = np.loadtxt(open('data.csv', 'r'), delimiter = ',')
data_labels = np.loadtxt(open('dataLabels.csv', 'r'))
data_matrix = data_matrix.tolist()
data_labels = data_labels.tolist()
print("Loading Data Finished!\n")

# 如果神经网络不存在，则用5000个数据进行训练
print("Startup Training ... ")
train_indices, test_indices = train_test_split(range(5000))
# 这里不知道出现了什么问题，当使用train_test_split时，效果较好，但训练数据增加到5000整时效果变为0.1(均预测为9)
# train_indices = list(range(5000))
test_indices = list(range(5000))
nn = OCRNeuralNetwork(HIDDEN_NODE_COUNT, data_matrix, data_labels, train_indices, False, False)
score = test(data_matrix, data_labels, test_indices, nn)
print("Startup Training Finished! Score on training set = %.3f\n" % score)

class JSONHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(s):
        response_code = 200
        response = ""
        var_len = int(s.headers.get('Content-Length'))
        content = s.rfile.read(var_len)
        payload = json.loads(content.decode())

        if payload.get('train'):
            nn.train(payload['trainArray'])
            nn.save()
        elif payload.get('predict'):
            try:
                response = {'result': nn.predict(payload['image'])}
            except:
                response_code = 500 # Server Error
        else:
            response_code = 400 # Bad Request

        s.send_response(response_code)
        s.send_header("Content-type", "application/json")
        s.send_header("Access-Control-Allow-Origin", "*")
        s.end_headers()
        if response:
            s.wfile.write(json.dumps(response).encode())
        return


if __name__ == '__main__':
    server_class = http.server.HTTPServer;
    httpd = server_class((HOST_NAME, PORT), JSONHandler)

    try:
        print("Server start to listen ... ")
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    else:
        print("Unexpected server exception occurred.")
    finally:
        httpd.server_close()