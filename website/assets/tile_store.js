const sheetTiles=(()=>{
  const stores=new Map(),ready=new Map();
  const split=path=>{const k=path.lastIndexOf('/');return [path.slice(0,k+1),path.slice(k+1)]};
  function store(folder){
    let s=stores.get(folder);
    if(!s){s=fetch(folder+'tiles.index',{cache:'no-cache'}).then(r=>r.ok?r.arrayBuffer():Promise.reject(new Error('no index'))).then(buffer=>{
      const size=new DataView(buffer).getUint32(8,true),header=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,12,size))),start=Math.ceil((12+size)/8)*8;
      const found={header,offsets:new Float64Array(buffer,start,header.cells),lengths:new Uint32Array(buffer,start+8*header.cells,header.cells),pack:folder+header.pack+'?v=1'};
      ready.set(folder,found);return found});
      stores.set(folder,s)}
    return s}
  return {
    // Reload the index once if publication replaced its pack.
    async blob(L,level,c,r,again=false){
      const [folder,name]=split(L.path),s=await store(folder),entry=s.header.layers[name];
      if(!entry||level>=entry.grid.length)return null;
      const [columns,rows]=entry.grid[level],[c0,r0]=entry.origin?entry.origin[level]:[0,0];
      if(c<c0||r<r0||c-c0>=columns||r-r0>=rows)return null;
      let k=entry.first+(r-r0)*columns+c-c0;
      for(let l=0;l<level;l++)k+=entry.grid[l][0]*entry.grid[l][1];
      const length=s.lengths[k],first=s.offsets[k];
      if(!length)return null;
      const response=await fetch(s.pack,{headers:{Range:`bytes=${first}-${first+length-1}`}});
      if(response.status===404&&!again){stores.delete(folder);ready.delete(folder);return this.blob(L,level,c,r,true)}
      if(response.status!==206)throw new Error('no byte range');
      const bytes=await response.arrayBuffer();
      return new Blob([bytes],{type:new Uint8Array(bytes,0,1)[0]===0x89?'image/png':'image/jpeg'})},
    // [low, gain] of a layer's one linear stretch as fractions of full scale; [0, 1] for a layer stored as shown.
    shown(L){
      const [folder,name]=split(L.path),s=ready.get(folder),entry=s&&s.header.layers[name];
      return entry&&entry.raw?[entry.low/255,255/(entry.high-entry.low)]:[0,1]}};
})();
