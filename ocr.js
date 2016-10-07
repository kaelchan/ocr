var ocrDemo = {
    CANVAS_WIDTH: 200,
    PIXEL_WIDTH: 10, // TRANSLATED_WIDTH = CANVAS_WIDTH / PIXEL_WIDTH
    TRANSLATED_WIDTH: 200 / 10, // CANVAS_WIDTH / PIXEL_WIDTH
    BATCH_SIZE: 5,

    // Server
    PORT: "8000",
    HOST: "http://localhost",

    // Colors
    BLACK: "#000000",
    BLUE: "#0000ff",

    trainArray: [],
    trainingRequestCount: 0,

    onLoadFunction: function() {
        this.resetCanvas();
    },

    resetCanvas: function() {
        var canvas = document.getElementById('canvas');
        var ctx = canvas.getContext('2d');

        this.data = [];
        ctx.fillStyle = this.BLACK;
        ctx.fillRect(0, 0, this.CANVAS_WIDTH, this.CANVAS_WIDTH);
        var matrixSize = this.TRANSLATED_WIDTH * this.TRANSLATED_WIDTH; // = 400
        while (matrixSize--)
            this.data.push(0);
        this.drawGrid(ctx);

        canvas.onmousemove = function(event) {this.onMouseMove(event, ctx, canvas)}.bind(this);
        canvas.onmousedown = function(event) {this.onMouseDown(event, ctx, canvas)}.bind(this);
        canvas.onmouseup   = function(event) {this.onMouseUp(event, ctx)}.bind(this);

        console.log("Reset Finished!")
    },

    drawGrid: function(ctx) {
        for (var x = this.PIXEL_WIDTH, y = this.PIXEL_WIDTH;
                 x < this.CANVAS_WIDTH;
                 x += this.PIXEL_WIDTH, y += this.PIXEL_WIDTH) {
            ctx.strokeStyle = this.BLUE;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, this.CANVAS_WIDTH);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(this.CANVAS_WIDTH, y);
            ctx.stroke();
        }
    },

    onMouseMove: function(event, ctx, canvas) {
        if (canvas.isDrawing) {
            this.fillSquare(event, ctx, canvas);
        }
    },

    onMouseDown: function(event, ctx, canvas) {
        canvas.isDrawing = true;
        this.fillSquare(event, ctx, canvas);
    },

    onMouseUp: function(event) {
        canvas.isDrawing = false;
    },

    fillSquare: function(event, ctx, canvas) {
        var x = event.clientX - canvas.offsetLeft;
        var y = event.clientY - canvas.offsetTop;
        var xPixel = Math.floor( x / this.PIXEL_WIDTH );
        var yPixel = Math.floor( y / this.PIXEL_WIDTH );
        this.data[(xPixel - 1) * this.TRANSLATED_WIDTH + yPixel - 1] = 1; // 为什么要减一,查询math.floor

        ctx.fillStyle = '#ffffff';
        ctx.fillRect(xPixel * this.PIXEL_WIDTH, yPixel * this.PIXEL_WIDTH, 
                     this.PIXEL_WIDTH, this.PIXEL_WIDTH);
    },

    train: function() {
        var digitVal = document.getElementById("digit").value;
        if (!digitVal || this.data.indexOf(1) < 0) {
            alert("Please type and draw a digit value in order to train the network!");
            return;
        }
        this.trainArray.push({"y0": this.data, "label": parseInt(digitVal)});
        this.trainingRequestCount++;

        // Send the trainning batch to the server
        if (this.trainingRequestCount == this.BATCH_SIZE) {
            alert("Sending training data to server ... ");
            var json = {
                trainArray: this.trainArray,
                train: true
            };

            this.sendData(json);
            this.trainingRequestCount = 0;
            this.trainArray = [];
        }
    },

    test: function() {
        if (this.data.indexOf(1) < 0) {
            alert("Please draw a something first!");
            return;
        }
        var json = {
            image: this.data,
            predict: true
        };
        this.sendData(json);
    },

    receiveResponse: function(xmlHttp) {
        if (xmlHttp.status != 200) { // http状态消息， 200表示成功
            alert("Server returned status" + xmlHttp.status);
            return;
        }
        var responseJSON = JSON.parse(xmlHttp.response);
        alert("I think you wrote a \'" + responseJSON.result + "\'");
    },

    onError: function(event) {
        alert("Error while connecting to server!" + event.target.statusText); // event.target表示触发事件的对象
    },

    sendData: function(json) {
        var xmlHttp = new XMLHttpRequest();
        xmlHttp.open("POST", this.HOST + ":" + this.PORT, true);
        xmlHttp.onload = function() {this.receiveResponse(xmlHttp)}.bind(this);
        xmlHttp.onerror = function() {this.onError(xmlHttp)}.bind(this);
        var msg = JSON.stringify(json);
        console.log(msg.length);
        console.log(msg);
        xmlHttp.setRequestHeader('Content-length', msg.length);
        xmlHttp.setRequestHeader('Connection', 'close'); // 指定连接在发送完这次信息后关闭，对应的是'keep-alive'
        xmlHttp.send(msg);
    },
}