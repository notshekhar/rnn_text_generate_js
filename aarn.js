class Matrix {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.data = Array(this.rows).fill().map(() => Array(this.cols).fill(0));
  }
  static fromArray(arr) {
    return new Matrix(arr.length, 1).map((e, i) => arr[i]);
  }

  static subtract(a, b) {
    if(a.rows!==b.rows || a.cols!==b.cols){
      console.log('Columns and Rows of A must match Columns and Rows of B.');
      return;
    }

    // Return a new Matrix a-b
    return new Matrix(a.rows, a.cols)
      .map((_, i, j) => a.data[i][j] - b.data[i][j]);
  }

  toArray() {
    let arr = [];
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        arr.push(this.data[i][j]);
      }
    }
    return arr;
  }

  randomize() {
    return this.map(e => Math.random() * 2 - 1);
  }

  add(n) {
    if (n instanceof Matrix) {
      if(this.rows!==n.rows || this.cols!==n.cols){
        console.log('Columns and Rows of A must match Columns and Rows of B.');
        return;
      }
        return this.map((e, i, j) => e + n.data[i][j]);
    } else {
      return this.map(e => e + n);
    }
  }
  static add(a, b){
    if(a.rows!==b.rows || a.cols!==b.cols){
      console.log('Columns and Rows of A must match Columns and Rows of B.');
      return;
    }
    // Return a new Matrix a-b
    return new Matrix(a.rows, a.cols)
    .map((_, i, j) => a.data[i][j] + b.data[i][j]);
  }
  static transpose(matrix) {
    return new Matrix(matrix.cols, matrix.rows)
      .map((_, i, j) => matrix.data[j][i]);
  }

  static multiply(a, b) {
    // Matrix product
    if (a.cols !== b.rows) {
      console.log('Columns of A must match rows of B.')
      return;
    }

    return new Matrix(a.rows, b.cols)
      .map((e, i, j) => {
        // Dot product of values in col
        let sum = 0;
        for (let k = 0; k < a.cols; k++) {
          sum += a.data[i][k] * b.data[k][j];
        }
        return sum;
      });
  }


  multiply(n) {
    if (n instanceof Matrix) {
      if(this.rows!==n.rows || this.cols!==n.cols){
        console.log('Columns and Rows of A must match Columns and Rows of B.');
        return;
      }
      // hadamard product
      return this.map((e, i, j) => e * n.data[i][j]);
    } else {
      // Scalar product
      return this.map(e => e * n);
    }
  }

  map(func) {
    // Apply a function to every element of matrix
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        let val = this.data[i][j];
        this.data[i][j] = func(val, i, j);
      }
    }
    return this;
  }

  static map(matrix, func) {
    // Apply a function to every element of matrix
    return new Matrix(matrix.rows, matrix.cols)
      .map((e, i, j) => func(matrix.data[i][j], i, j));
  }

  print() {
    console.table(this.data);
    return this;
  }

  serialize() {
    return JSON.stringify(this);
  }

  static deserialize(data) {
    if(typeof data == 'string')
    {
      data = JSON.parse(data);
    }
    let matrix = new Matrix(data.rows, data.cols);
    matrix.data = data.data;
    return matrix;
  }
}

if (typeof module !== 'undefined') {
  module.exports = Matrix;
}

class math {
  constructor() {

  }
  static mouse(r){
    document.onmousemove = function(e){
      r(e.clientX, e.clientY)
    }
  }
  static sort(arr){
    let n = 0
    for (let i = n; i < arr.length; i++) {
      let smallest = arr[n]
      for(let j = n; j<arr.length; j++){
        if(arr[j]<smallest){
          smallest = arr[j]
          arr[j] = arr[n]
          arr[n] = smallest
        }
      }
      n+=1
    }
    // console.log(arr);
    return arr
  }

  static sigma(a){
    let sigma = new Array(a-1);
    if(a>2){
      sigma[0] = 1;
      sigma[1] = 1;
      for(let i = 2; i<sigma.length; i++){
        sigma[i] = sigma[i-1] + sigma[i-2];

      }
      // console.log(sigma[sigma.length-1]);
      return sigma[sigma.length-1];
    }else{
      // console.log(1);
      return 1;
    }
  }
  static rgba(arr){
    let r=[];
    let g=[];
    let b =[];
    let a = [];

    for (let i = 0; i < arr.length; i+=4) {
      r.push(arr[i]);
      g.push(arr[i+1]);
      b.push(arr[i+2]);
      a.push(arr[i+3]);
    }
    return {"red": r, "green": g, "blue": b, "alpha": a};
  }
  static cto(pixels, maxVal){
    let p=[];
    for (let i = 0; i < pixels.length; i++) {
      p.push(pixels[i]/maxVal);
    }
    return p;
  }
  static findmax(arr){
    let n = 0;
    let m;
    for (var i = 0; i < arr.length; i++) {
      if(arr[i]>n){
        m = i;
        n = arr[i];
      }
    }
    return m;
  }

  static add(a,b){
    return a+b;
  }
  static subtract(a,b){
    return a-b;
  }
  static multiply(a,b){
    return a*b;
  }
  static divide(a,b){
    return a/b;
  }

  static sumArray(arr){
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
      sum+=arr[i];
    }
    return sum;
  }
  static turn(arr){
    for(let i=0; i<arr.length; i++){
      if(arr[i]>0.5){
        arr[i]=1;
      }else {
        arr[i]=0;
      }
    }
    return arr;
  }
  static pixr(url, a, b, func){
    let i = new Image(a,b);
    i.src = url;
    let canva = document.createElement("canvas");
    canva.height = i.height;
    canva.width = i.width;
    let ctx = canva.getContext("2d");
    let red=[];
    let green=[];
    let blue=[];
    let alpha=[];
    i.onload = function(){
      ctx.drawImage(i, 0, 0, i.width, i.height);
      // console.log("url",canva.toDataURL());
      let d = ctx.getImageData(0,0,canva.width,canva.height).data;
      for(let i=0; i<d.length; i+=4){
        red.push(d[i]);
        green.push(d[i+1]);
        blue.push(d[i+2]);
        alpha.push(d[i+3]);
      }
      // console.log({"red":red, "green": green, "blue": blue, "alpha": alpha});
      func({"red":red, "green": green, "blue": blue, "alpha": alpha})
    }
    // document.body.append(canva);
  }
  static toDataURL(url, a, b){
    let i = new Image(a,b);
    i.src = url;
    let canva = document.createElement("canvas");
    canva.height = i.height;
    canva.width = i.width;
    let ctx = canva.getContext("2d");
    i.onload = function(){
      ctx.drawImage(i, 0, 0, i.width, i.height);
      let data = canva.toDataURL();
      console.log(data);
      }
    }
    static encode(text){
      let arr = Array()
      arr = [" ","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z",0,1,2,3,4,5,6,7,8,9]
      let e ="e"
      for(let i=0; i<text.length; i++){
        e+='_'+JSON.stringify(arr.indexOf(text[i]))
      }
      let k = e.split("_")
      k.splice(0, 1)
      for(let j=0; j<k.length; j++){
        k[j] = parseInt(k[j])
      }
      return {"encoded": e, "codes": k}
    }
    static decode(e){
      let k = e.split("_")
      let arr = Array()
      arr = [" ","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z",0,1,2,3,4,5,6,7,8,9]
      let text = 'd_'
      for(let i=1; i<k.length; i++){
        text+=arr[k[i]]
      }
      return text
    }

}


function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function dsigmoid(y) {
  // return sigmoid(x) * (1s - sigmoid(x));
  return y * (1 - y);
}

// function tanh(x) {
//   var y = Math.tanh(x);
//   return y;
// }
//
// function dtanh(x) {
//   var y = 1 / (pow(Math.cosh(x), 2));
//   return y;
// }


class fnn {
  constructor(arr, lr) {
    this.neurons = [];
    this.weights = [];
    this.lr = lr || 0.01;
    let arrlen = arr.length;
    for(let i=0; i<arrlen; i++){
      this.neurons.push(arr[i]);
    }
    for(let i=0; i<arrlen-1; i++){
      let weight = new Matrix(this.neurons[i+1], this.neurons[i]);
      weight.randomize();
      this.weights.push(weight);
    }

  }
  predict(inputarr){
    let inputs = Matrix.fromArray(inputarr);
    let outputs = [];
    let weightlen = this.weights.length;
    for(let i=0; i<weightlen; i++){
      inputs = Matrix.multiply(this.weights[i], inputs);
      inputs.map(sigmoid);
      outputs.push(inputs);
    }
    return outputs;

  }
  query(arr){
  	let outputs = this.predict(arr);
  	let output = outputs[outputs.length-1].toArray();
  	return output;

  }

  learn(input, outputarr){
  	let inputs = Matrix.fromArray(input);
    let weightlen = this.weights.length;
    for(let i=0; i<weightlen; i++){
      inputs = Matrix.multiply(this.weights[i], inputs);
      inputs.map(sigmoid);
    }
    let output = inputs;
  	let answer = Matrix.fromArray(outputarr);
  	let err = Matrix.subtract(answer, output);
  	let errors = [];
  	for (var i = this.weights.length - 1; i >= 0; i--) {
  		errors.push(err);
  		err = Matrix.multiply(Matrix.transpose(this.weights[i]), err);
  	}
  	errors.reverse();
  	let outputs = [];
  	let inpout = [];
  	inpout.push(Matrix.fromArray(input));
  	let inp = Matrix.fromArray(input)
    for(let i=0; i<weightlen; i++){
      inp = Matrix.multiply(this.weights[i], inp);
      inp.map(sigmoid);
      outputs.push(inp);
      inpout.push(inp);
    }

   for (let i = 0; i < errors.length; i++) {
   	  let gradient = errors[i].multiply(Matrix.map(outputs[i], dsigmoid));
   	  let dweight = Matrix.multiply(gradient, Matrix.transpose(inpout[i]));
   	  dweight.multiply(this.lr);
      this.weights[i] = this.weights[i].add(dweight);
   }



  }
  setLearningRate(learn){
  	this.lr = learn;

  }


  download(filename){
    let arr = {
      "lr": this.lr,
      "neurons": this.neurons,
      "weights": this.weights
    }

    let datStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(arr));
    let downloadNode = document.createElement("a");
    downloadNode.setAttribute("href", datStr);
    downloadNode.setAttribute("download", filename + ".json");
    downloadNode.click();
    downloadNode.remove();

  }
  upload(weights){
    for(let i=0; i<this.weights.length; i++){
      for (let m = 0; m < this.weights[i].rows; m++) {
        for (let n = 0; n < this.weights[i].cols; n++) {
          this.weights[i].data[m][n] = weights[i].data[m][n];
        }
      }
    }

  }
  accuracy(input, arr){
    let answer = arr;
    let output = this.query(input);
    answer = math.findmax(answer);
    output = math.findmax(output);
    let score = [];
    if(answer == output){
      score.push(1);
    }else{
      score.push(0);
    }
    let accuracy = math.sumArray(score)/score.length*100;
    return accuracy;

  }

}


// Word Vectors
// Based on: https://github.com/turbomaze/word2vecjson
// declare wordVecs globally to store wordVec.json data
class Word2Vec {

  static magnitude(a) {
    let sum = a.reduce((sum, val) => {
      return sum + val*val;
    }, 0);
    return Math.sqrt(sum);
  }

  // Cosine similarity!
  static distance(v1, v2) {
    // Check if v1 or v2 is a string then grab vector?
    // let v1 = wordVecs[word1];
    // let v2 = wordVecs[word2];

    let sum = v1.reduce((sum, a, i) => {
      return sum + a * v2[i];
    }, 0);
    return sum / (this.magnitude(v1) * this.magnitude(v2)); //magnitude is 1 for all feature vectors
  }

  // Add two word vectors
  static add(v1, v2) {
    return v1.map((a, i) => a + v2[i]);
  }

  // Subtract two word vectors
  static subtract(v1, v2) {
    return v1.map((a, i) => a - v2[i]);
  }

  // Average of two word vectors
  static average(v1, v2) {
    return v1.map((a, i) => (a + v2[i]) * 0.5);
  }

  static nearest(wordVecs, word, n=10) {
    let vec;
    if (word instanceof Array) {
      vec = word;
    } else {
      if (!wordVecs[word]) {
        return undefined;
      } else {
        vec = wordVecs[word];
      }
    }
    let words = [];
    let keys = Object.keys(wordVecs);
    for (let i = 0; i < keys.length; i++) {
      let key = keys[i];
      let d = this.distance(vec, wordVecs[key]);
      words.push({word: key, distance: d});
    }
    words.sort((a, b) => {
      return b.distance - a.distance;
    });
    return words.slice(0, n);
  }
}
