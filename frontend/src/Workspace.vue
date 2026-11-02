<script setup lang="ts">
import {ref,onMounted,onUnmounted} from 'vue'
import {api,post,request} from './api'
type Box={class_name:string,confidence:number,bbox_xyxy:number[]}
type Task={id:string,status:string,detections?:Box[],original_url:string,image_url:string}
type Citation={title:string,location:string,url:string}
type Turn={client_message_id:string,message:string,answer:string,citations?:Citation[],task_id?:string,operation_id?:string}
type Op={id:string,title:string,status:string,diff:string}
const page=ref('图片检测'), busy=ref(false), error=ref('')
const photo=ref<File>(), conf=ref(0.25), task=ref<Task>(), original=ref(''), marked=ref('')
const threads=ref<{thread_id:string,title:string}[]>([]), thread=ref(''), turns=ref<Turn[]>([])
const question=ref(''), attachment=ref<File>(), contextTask=ref(''), fileInput=ref<HTMLInputElement>()
const docs=ref<{id:string,title:string,version:number}[]>([]), versions=ref<{id:string,number:number}[]>([])
const docId=ref(''), content=ref(''), op=ref<Op>()
let alive=true
let pendingChat: {thread:string, message:string, linked:string, file?:File, attachmentId?:string, id:string} | undefined
let pendingDetection: {file:File, conf:number, key:string} | undefined
function clearImages(){for(const u of [original.value,marked.value]) if(u) URL.revokeObjectURL(u); original.value=''; marked.value=''}
onUnmounted(()=>{alive=false;clearImages()})
async function action(fn:()=>Promise<void>){
  if(busy.value)return
  busy.value=true; error.value=''
  try{await fn()}catch(e){error.value=String(e)}finally{busy.value=false}
}
function picked(e:Event){return (e.target as HTMLInputElement).files?.[0]}
async function showTask(id:string){
  clearImages()
  for(let i=0;i<90 && alive;i++){
    const t:Task=await api('/api/detections/'+id); task.value=t
    if(t.status==='succeeded'){
      const blobs=await Promise.all([t.original_url,t.image_url].map(async url=>(await request(url)).blob()))
      if(alive){original.value=URL.createObjectURL(blobs[0]); marked.value=URL.createObjectURL(blobs[1])}
      return
    }
    if(!['pending','running'].includes(t.status))throw new Error('检测状态：'+t.status)
    await new Promise(r=>setTimeout(r,1000))
  }
}
async function detect(){
  if(!photo.value)throw new Error('请选择图片')
  const body=new FormData();body.append('file',photo.value);body.append('conf',String(conf.value))
  clearImages();task.value=undefined
  if(!pendingDetection || pendingDetection.file!==photo.value || pendingDetection.conf!==conf.value)
    pendingDetection={file:photo.value,conf:conf.value,key:crypto.randomUUID()}
  const t=await api('/api/detections',{method:'POST',body,headers:{'Idempotency-Key':pendingDetection.key}})
  pendingDetection=undefined
  await showTask(t.id)
}
async function refresh(){threads.value=await api('/api/conversations');docs.value=await api('/api/documents')}
async function choose(id:string){
  thread.value=id; turns.value=[]; op.value=undefined; contextTask.value=''; content.value=''
  if(id)turns.value=await api('/api/conversations/'+id+'/messages')
}
async function newThread(){const r=await api('/api/conversations',post());await refresh();await choose(r.thread_id)}
async function send(){
  if(!question.value.trim())throw new Error('请输入请求')
  const linked=contextTask.value
  if(!thread.value)await newThread()
  contextTask.value=linked
  if(!pendingChat || pendingChat.thread!==thread.value || pendingChat.message!==question.value ||
      pendingChat.file!==attachment.value || pendingChat.linked!==linked){
    pendingChat={thread:thread.value,message:question.value,linked,file:attachment.value,id:crypto.randomUUID()}
  }
  if(attachment.value && !pendingChat.attachmentId){
    const body=new FormData();body.append('file',attachment.value)
    body.append('purpose',attachment.value.name.toLowerCase().endsWith('.md')?'knowledge':'image')
    pendingChat.attachmentId=(await api('/api/files',{method:'POST',body})).file_id
  }
  const reply=await api('/api/chat',post({thread_id:thread.value,client_message_id:pendingChat.id,
    message:question.value,attachment_ids:pendingChat.attachmentId?[pendingChat.attachmentId]:[],detection_task_id:linked||null}))
  pendingChat=undefined
  question.value='';attachment.value=undefined;if(fileInput.value)fileInput.value.value=''
  await choose(thread.value);await refresh()
  if(reply.operation_id)await preview(reply.operation_id)
}
async function preview(id:string){op.value=await api('/api/knowledge/operations/'+id)}
async function resolve(confirm:boolean){
  if(!op.value)return
  op.value=await api('/api/knowledge/operations/'+op.value.id+'/'+(confirm?'confirm':'cancel'),post())
  await refresh()
}
async function showVersions(id:string){docId.value=id;versions.value=await api('/api/documents/'+id+'/versions');content.value=''}
async function read(url:string){const r=await api(url);content.value=r.title+' · v'+r.number+'\n\n'+r.content}
function switchPage(p:string){page.value=p; content.value='';error.value=''}
onMounted(()=>action(async()=>{await refresh();if(threads.value.length)await choose(threads.value[0].thread_id)}))
</script>
<template><main><nav><button v-for="p in ['图片检测','智能问答','知识管理']" :class="{active:page===p}" :disabled="busy" @click="switchPage(p)">{{p}}</button></nav>
<h1>{{page}}</h1><p role="status" v-if="busy">正在处理…</p><p role="alert" v-if="error">{{error}}</p>
<section v-if="page==='图片检测'"><label>红外图片<input type="file" accept="image/jpeg,image/png" @change="photo=picked($event)"></label>
<label>置信度阈值<input v-model.number="conf" type="number" min="0" max="1" step="0.05"></label>
<button class="primary" :disabled="busy" @click="action(detect)">开始检测</button>
<template v-if="task"><p>{{task.status}} · {{task.id}}</p><div class="images"><figure v-if="original"><img :src="original"><figcaption>原图</figcaption></figure><figure v-if="marked"><img :src="marked"><figcaption>预测图</figcaption></figure></div>
<button :disabled="busy" @click="switchPage('智能问答');contextTask=task.id">追问本次检测</button>
<table><thead><tr><th>编号</th><th>类别</th><th>置信度</th><th>原图坐标 xyxy</th></tr></thead><tbody><tr v-for="(b,i) in task.detections"><td>{{i+1}}</td><td>{{b.class_name}}</td><td>{{b.confidence.toFixed(3)}}</td><td>{{b.bbox_xyxy.map(Math.round).join(', ')}}</td></tr></tbody></table></template></section>
<template v-else>
<section v-if="page==='知识管理'"><h2>公共文档</h2><p v-if="!docs.length">尚无已发布文档。附上 Markdown 并输入导入要求即可发起提议。</p>
<button v-for="d in docs" :disabled="busy" @click="action(()=>showVersions(d.id))">{{d.title}} · v{{d.version}}</button>
<div><button v-for="v in versions" :disabled="busy" @click="action(()=>read('/api/documents/'+docId+'/versions/'+v.id))">查看版本 {{v.number}}</button></div></section>
<section><label>当前会话<select :value="thread" :disabled="busy" @change="action(()=>choose(($event.target as HTMLSelectElement).value))"><option value="">选择会话</option><option v-for="t in threads" :value="t.thread_id">{{t.title}}</option></select></label><button :disabled="busy" @click="action(newThread)">新会话</button>
<article v-for="t in turns" :key="t.client_message_id"><b>{{t.message}}</b><p>{{t.answer}}</p><button v-for="c in t.citations" :disabled="busy" @click="action(()=>read(c.url))">{{c.title}} · {{c.location}}</button>
<button v-if="t.task_id" :disabled="busy" @click="switchPage('图片检测');action(()=>showTask(t.task_id!))">查看检测</button><button v-if="t.operation_id" :disabled="busy" @click="action(()=>preview(t.operation_id!))">查看变更</button></article>
<p v-if="contextTask">关联检测 {{contextTask}}</p><label>指令<textarea v-model="question" :placeholder="page==='知识管理'?'例：将附件完整导入公共知识库':'例：上次检测识别出了哪些类别？'"></textarea></label>
<input ref="fileInput" type="file" accept=".md,.jpg,.jpeg,.png" @change="attachment=picked($event)"><button class="primary" :disabled="busy" @click="action(send)">发送</button></section>
<section v-if="op"><h2>{{op.title}} · {{op.status}}</h2><pre>{{op.diff}}</pre><template v-if="op.status==='pending'"><button :disabled="busy" @click="action(()=>resolve(true))">确认发布</button><button :disabled="busy" @click="action(()=>resolve(false))">取消</button></template></section>
<section v-if="content"><h2>版本原文</h2><pre>{{content}}</pre></section>
</template></main></template>
