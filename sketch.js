var xs_real = [] //empty arrays for dataset from mouse locations
var ys_real =[]
let m ,b ;

const lrate = 0.5;
const optmz = tf.train.sgd(lrate)

function setup(){
  var canvas =createCanvas(window.innerWidth, window.innerHeight);
  canvas.parent('canvas')
  background('black');
  m = tf.variable(tf.scalar(random(10)));
  b = tf.variable(tf.scalar(random(10)));
}

const loss = (pred, label) => pred.sub(label).square().mean();


const predict = (x) =>
{
  const tfxs = tf.tensor1d(x)
  return tfxs.mul(m).add(b)           // y= mx+c
}

function mousePressed ()
{
  x = map(mouseX , 0 , windowWidth , 0 , 1 ) //getting mouse x and converting in range 0-1
  y = map(mouseY, 0, windowHeight, 0, 1) // getting mouse y and converting in range 0-1
  xs_real.push(x)
  ys_real.push(y) 
  console.log(x,y)
  
}  
function windowResized() {
  setup()
}
function draw(){

  background(0)

 if(xs_real.length>1){optmz.minimize(() => loss(predict(xs_real), tf.tensor1d(ys_real))); }   
  stroke('red')
  strokeWeight(10)
  for(let i = 0;i< xs_real.length;i++)
  {
    let px = map(xs_real[i] , 0 , 1 , 0 , windowWidth)
    let py = map(ys_real[i], 0, 1,0,windowHeight)

    point(px,py)
  }

  tf.tidy(()=>{

  const xs = [0,1]
  let ys = predict(xs)
  let x1 = map(xs[0] , 0,1 ,0 ,windowWidth)
  let x2 = map(xs[1] , 0 ,1,0 ,windowWidth)
  let liney = ys.dataSync()  
  let y1 = map(liney[0],0,1,0,windowHeight)
  let y2 = map(liney[1], 0,1, 0, windowHeight)

  strokeWeight(4)

  line(x1, y1, x2, y2)
  });
  
}
