
//Recurrent Neural Network
class rnn{
  constructor(nIn, nHidden, nOut, time, lr){
    this.input = nIn
    this.hidden = nHidden
    this.output = nOut
    this.time = time
    this.lr = lr
    this.U = new Matrix(nHidden, nIn).randomize() //weights b/w hidden and input
    this.V = new Matrix(nOut, nHidden).randomize() //weights b/w output and hidden
    this.H = new Matrix(nHidden, nHidden) //hidden state weights
  }
  feedforward(x){
    let timeLength = x.length
    let input = new Array()
    for(let i=0; i<x.length; i++){
      let m = [0]
      m[0] = x[i]
      input.push(Matrix.fromArray(m))
    }
    let s = new Array()
    s.push(new Matrix(this.hidden, 1))
    let u = new Array()
    let y = new Array()
    let v = new Array()
    for(let t=0; t<timeLength; t++){
      u.push(Matrix.add(Matrix.multiply(this.U, input[t]), Matrix.multiply(this.H, s[t])))
      s.push(u[t].map(sigmoid))
      v.push(Matrix.multiply(this.V, s[t+1]))
      y.push(v[t].map(sigmoid))
    }
    return{
      u: u,
      s: s,
      v: v,
      y: y
    }
  }
  query(x){
    let o = this.feedforward(x)
    return o.y
  }
  train(x, o){
    //magical codes will go here
  }
  generateText(x,tp, func){
    x = x.toLowerCase()
    x = x.split(' ')
    tp = tp || 10
    let r = new XMLHttpRequest()
    r.open('GET', "wordvecs1000.json")
    let U  = this.U
    let H = this.H
    let V = this.V
    let hidden = this.hidden
    r.onload = function(){
      let vectors = JSON.parse(r.responseText).vectors
      let timeLength = x.length
      let input = new Array()
      for(let i=0; i<x.length; i++){
        let vector = vectors[x[i]]
        let mat = Matrix.fromArray(vector)
        input.push(mat)
      }
      let s = new Array()
      s.push(new Matrix(hidden, 1))
      let u = new Array()
      let y = new Array()
      let v = new Array()
      if(tp>timeLength){
        for(let t=0; t<tp; t++){
          if(t > timeLength-1){
            u.push(Matrix.add(Matrix.multiply(U, y[t-1]), Matrix.multiply(H, s[t])))
          }else{
            u.push(Matrix.add(Matrix.multiply(U, input[t]), Matrix.multiply(H, s[t])))
          }
          s.push(u[t].map(sigmoid))
          v.push(Matrix.multiply(V, s[t+1]))
          y.push(v[t].map(sigmoid))
        }
      }else{
        for(let t=0; t<tp; t++){
          u.push(Matrix.add(Matrix.multiply(U, input[t]), Matrix.multiply(H, s[t])))
          s.push(u[t].map(sigmoid))
          v.push(Matrix.multiply(V, s[t+1]))
          y.push(v[t].map(sigmoid))
        }
      }
      let output = new Array()
      for(let i=0; i<y.length; i++){
        output.push(y[i].toArray())
      }
      let words = new Array()
      for(let i=0; i<output.length; i++){
        words.push(Word2Vec.nearest(vectors ,output[i], 1))
      }
      func(words)
    }
    r.send()
  }

}
