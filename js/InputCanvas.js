// add canvas to draw on
class InputCanvas {
  constructor(args){
    this.canvas = document.getElementById(args.canvas)
    this.ctx = null;
    this.width = args.width;
    this.height = args.height;
    this.hoverPos = [[0, 0], [0, 0]]; // [[prevX, prevY], [x, y]]
    this.dragging = false;
    this.lineWidth = 30; // todo make relative
    this.bgColor = args.bgColor;
    this.strokeStyle = args.strokeStyle;
    this.imageUrl = new Image();
  }
  // setup canvas
  init(){
    this.ctx = this.canvas.getContext('2d');
    // TODO set w/h relative to parent element
    this.canvas.width = this.width;
    this.canvas.height = this.height;
    this.ctx.fillStyle = this.bgColor;
    this.ctx.fillRect(0, 0, this.width, this.height);
    this.canvas.onmousemove = e => this.hover(e); 
    this.canvas.onmousedown = e => this.mouseDown(e);
    this.canvas.onmouseup = e => this.mouseUp(e);
    window.addEventListener('resize', () => this.resize());
  }
  // monitor mouse x, y. keep prev and current positions in hoverPos, draw if dragging
  hover(e){
    if (this.dragging) { 
        this.draw()
      }
      this.hoverPos.shift()
      this.hoverPos.push([e.offsetX, e.offsetY])
  }
  // reset hoverPos to current position and flag dragging
  mouseDown(e){ 
    this.hoverPos = [[e.offsetX, e.offsetY], [e.offsetX, e.offsetY]]
    this.dragging = true;
  }
  // flag dragging
  mouseUp(){
    this.dragging = false;
  }
  // draw from prev X/Y to current X/Y linecap = 'round' is the secret
  draw(){
    this.ctx.strokeStyle = this.strokeStyle;
    this.ctx.lineWidth = this.lineWidth;
    this.ctx.lineCap = 'round';
    this.ctx.beginPath();
    this.ctx.moveTo(...this.hoverPos[0]);
    this.ctx.lineTo(...this.hoverPos[1]);
    this.ctx.stroke();
    this.imageUrl.src = this.canvas.toDataURL('image/png');
    console.log(this.imageUrl)
  }
  // reset 
  clear() {
    this.ctx.fillStyle = this.bgColor;
    this.ctx.fillRect(0, 0, this.width, this.height);
  }
  // TODO resize relative to paraent element
  resize(){
    console.log('resize')
  }
  // return a 28 x 28 img of canvas
  castToImage(){
    const imageTensor = tf.tidy(() => {
    const scaleImage = tf.browser.fromPixels(this.imageUrl, 1)
    const shape = tf.image.resizeBilinear(scaleImage, [28, 28]);
    return shape.expandDims(0);
  });
  return imageTensor
  }
}