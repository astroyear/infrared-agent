<script setup lang="ts">
import {ref,onMounted,onUnmounted} from 'vue'
import {api,post} from './api'
import Workspace from './Workspace.vue'
const name=ref(''), password=ref(''), account=ref(''), message=ref(''), busy=ref(false)
function logout(){sessionStorage.removeItem('token'); account.value=''; password.value=''; message.value=''}
async function auth(register:boolean){
  busy.value=true; message.value=''
  try {
    const body={username:name.value,password:password.value}
    if(register) await api('/api/auth/register',post(body))
    const result=await api('/api/auth/login',post(body))
    sessionStorage.setItem('token',result.access_token)
    account.value=result.user.username; password.value=''
  }catch(e){message.value=String(e)}finally{busy.value=false}
}
onMounted(async()=>{
  window.addEventListener('session-expired',logout)
  if(sessionStorage.getItem('token'))try{account.value=(await api('/api/auth/me')).username}catch(e){message.value=String(e)}
})
onUnmounted(()=>window.removeEventListener('session-expired',logout))
</script>
<template>
<header><b>红外检测工作台</b><span v-if="account">{{account}}　<button @click="logout">退出</button></span></header>
<main v-if="!account" class="login"><small>INFRARED / UAV</small><h1>进入检测工作台</h1>
<form @submit.prevent="auth(false)"><label>用户名<input required v-model="name" autocomplete="username"></label>
<label>密码<input required type="password" v-model="password" autocomplete="current-password"></label>
<button class="primary" :disabled="busy">登录</button> <button type="button" :disabled="busy" @click="auth(true)">注册并登录</button></form><p role="alert">{{message}}</p></main>
<Workspace v-else :key="account"/>
</template>