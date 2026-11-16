if(typeof OffscreenCanvas==='undefined')throw new Error('Worker canvas unavailable');
const canvas=new OffscreenCanvas(512,512),context=canvas.getContext('2d',{willReadFrequently:true});
self.onmessage=async({data:{id,blob}})=>{
  let image;
  try{
    image=await createImageBitmap(blob,{colorSpaceConversion:'none'});
    const {width,height}=image;
    if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height}
    context.clearRect(0,0,width,height);context.drawImage(image,0,0);
    const rgba=context.getImageData(0,0,width,height).data,pixels=new Uint8Array(width*height);
    for(let i=0;i<pixels.length;i++)pixels[i]=rgba[i*4];
    self.postMessage({id,width,height,data:pixels},[pixels.buffer]);
  }catch(error){self.postMessage({id,error:String(error)})}
  finally{if(image)image.close()}
};
