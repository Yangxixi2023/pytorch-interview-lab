import {codeBlock} from './problem-view.js';

const esc = value => String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const $ = selector => document.querySelector(selector);

export function initStudy({api,problems,openProblem,showMain,toast}) {
  let mode='reference',selected='ppo_loss',filter='',category='全部',questions=[],code='',generation=0;
  const list=[{id:'tensor-guide',title:'张量变换与 einops 完整教程',category:'张量基础'},...problems];

  function shell() {
    showMain(false);
    $('#bank').hidden=true;
    $('#library').hidden=false;
    document.querySelectorAll('header nav button').forEach(b=>b.classList.remove('active'));
    $(`#${mode}-nav`).classList.add('active');
    $('#library').classList.toggle('qa-mode',mode==='interview');
    $('#library-title').textContent=mode==='reference'?`全部 ${problems.length} 题参考代码`:'训练岗位面试问答';
    $('#library-search').placeholder=mode==='reference'?'搜索参考代码…':'搜索问答、知识点…';
    $('#library-search').value=filter;
    const categories=mode==='reference'?['全部',...new Set(list.map(p=>p.category))]:['全部','机器学习','预训练','后训练'];
    $('#library-filter').innerHTML=categories.map(c=>`<option ${c===category?'selected':''}>${esc(c)}</option>`).join('');
    renderList();
  }

  function renderList() {
    if(mode==='interview') {
      $('#library-menu').innerHTML='<div class="study-side-note">先口述，再展开答案。<br>每题包含要点、追问、易错点和原始资料。<br><br>按训练岗位知识体系整理，不标注未经核实的公司面经。</div>';
      renderQuestions();
      return;
    }
    const rows=list.filter(p=>(category==='全部'||p.category===category)&&`${p.title} ${p.id} ${p.category}`.toLowerCase().includes(filter.toLowerCase()));
    $('#library-menu').innerHTML=rows.length?rows.map(p=>`<button data-reference="${p.id}" class="reference-item ${p.id===selected?'selected':''}"><span>${esc(p.title)}</span><small>${esc(p.category)}</small></button>`).join(''):'<p class="muted">没有匹配的参考实现。</p>';
  }

  async function renderReference(id) {
    const request=++generation;
    $('#library-content').innerHTML='<p class="muted">加载参考代码…</p>';
    try {
      if(id==='tensor-guide') {
        const guide=await api('/api/tensor-guide');
        if(request!==generation)return;
        code=guide.code;
        $('#library-content').innerHTML=`<div class="study-heading"><p class="section-kicker">TENSOR OPERATIONS</p><h1>张量变换与 einops</h1><p>完整覆盖 reshape / view、transpose / permute、contiguous、维度增减、广播、拼接、拆分与 where。</p></div><div class="study-actions"><button id="hide-reference">遮住代码</button><button id="copy-reference">复制代码</button><button id="download-reference">下载 .py</button><button class="primary" data-practice="tensor_heads">去练习拆头 →</button></div><div id="reference-code">${codeBlock(code)}</div>`;
      } else {
        const [p,solution]=await Promise.all([api('/api/problem/'+id),api('/api/solution/'+id)]);
        if(request!==generation)return;
        code=solution.code;
        const returnText=p.returns.map(r=>`${r.name}: ${r.type} ${r.shape}`).join('；');
        $('#library-content').innerHTML=`<div class="study-heading"><p class="section-kicker">REFERENCE / ${esc(p.category)}</p><h1>${esc(p.title)}</h1><p>读懂输入与返回，记住计算顺序，再进入练习独立写一遍。</p></div><div class="reference-signature"><strong>返回</strong><code>${esc(returnText)}</code></div><details class="reference-io"><summary>输入参数与返回要求</summary><h3>输入</h3>${p.inputs.map(v=>`<p><code>${esc(v.name)}</code> · ${esc(v.type)} · <code>${esc(v.shape)}</code><br>${esc(v.description)}</p>`).join('')}<h3>返回${p.returns.length>1?'（依次组成元组）':''}</h3>${p.returns.map(v=>`<p><code>${esc(v.name)}</code> · ${esc(v.type)} · <code>${esc(v.shape)}</code><br>${esc(v.description)}</p>`).join('')}</details><div class="study-actions"><button id="hide-reference">遮住代码</button><button id="copy-reference">复制代码</button><button id="download-reference">下载 .py</button><button class="primary" data-practice="${p.id}">去练习 / 默写 →</button></div><div id="reference-code">${codeBlock(code)}</div><details class="reference-rules"><summary>核对本题计算约定</summary><p>${esc(p.statement)}</p></details><p class="shape-note">进入练习会保留已有草稿。若想从空白模板重新默写，可以在编辑器点击“重置”。</p>`;
      }
    }catch(e){if(request===generation)$('#library-content').innerHTML=`<p class="failure">${esc(e.message)}</p>`}
  }

  function renderQuestions() {
    const rows=questions.filter(q=>(category==='全部'||q.category===category)&&`${q.title} ${q.answer} ${q.points.join(' ')}`.toLowerCase().includes(filter.toLowerCase()));
    $('#library-content').innerHTML=`<div class="study-heading"><p class="section-kicker">TRAINING INTERVIEW</p><h1>机器学习 · 预训练 · 后训练</h1><p>${rows.length} / ${questions.length} 道问答。回答按“结论 → 原理 → 边界”组织；先尝试回答，再展开核对。</p></div>${rows.map(q=>`<article class="qa-card"><div class="qa-meta">${q.id.toUpperCase()} · ${esc(q.category)}</div><h2>${esc(q.title)}</h2><details><summary>展开回答与追问</summary><div class="qa-answer"><h3>可以这样回答</h3><p>${esc(q.answer)}</p><h3>补充要点</h3><ul>${q.points.map(point=>`<li>${esc(point)}</li>`).join('')}</ul><div class="qa-followup"><strong>面试追问</strong><p>${esc(q.followup)}</p></div><p class="qa-pitfall"><strong>易错点：</strong>${esc(q.pitfall)}</p><div class="qa-sources">${q.sources.map(s=>`<a href="${esc(s.url)}" target="_blank" rel="noreferrer">${esc(s.title)} ↗</a>`).join('')}</div>${q.related.length?`<div class="qa-practice">${q.related.map(id=>`<button data-practice="${id}">练习 ${esc(problems.find(p=>p.id===id).title)} →</button>`).join('')}</div>`:''}</div></details></article>`).join('')}${!rows.length?'<p class="muted">没有匹配的问答。</p>':''}`;
  }

  $('#library-search').oninput=e=>{filter=e.target.value;renderList()};
  $('#library-filter').onchange=e=>{category=e.target.value;renderList()};
  $('#library-menu').onclick=e=>{const button=e.target.closest('[data-reference]');if(button)location.hash='reference:'+button.dataset.reference};
  $('#library-content').onclick=async e=>{
    const practice=e.target.closest('[data-practice]');
    if(practice){await openProblem(practice.dataset.practice);return}
    if(e.target.id==='hide-reference'){const block=$('#reference-code');block.hidden=!block.hidden;e.target.textContent=block.hidden?'显示参考代码':'遮住代码'}
    if(e.target.id==='copy-reference'){try{await navigator.clipboard.writeText(code);toast('参考代码已复制')}catch{toast('复制失败，请选中代码手动复制')}}
    if(e.target.id==='download-reference'){const url=URL.createObjectURL(new Blob([code],{type:'text/x-python'}));const link=document.createElement('a');link.href=url;link.download=selected+'.py';link.click();URL.revokeObjectURL(url)}
  };
  $('#reference-nav').onclick=()=>{location.hash='reference:'+selected};
  $('#interview-nav').onclick=()=>{location.hash='interview'};

  return {
    route(hash) {
      if(hash.startsWith('reference:')) {
        mode='reference';selected=hash.slice(10);filter='';category='全部';
        if(!list.some(p=>p.id===selected))selected='ppo_loss';
        shell();renderReference(selected);return true;
      }
      if(hash==='interview') {
        generation++;mode='interview';filter='';category='全部';shell();
        api('/api/interview').then(data=>{questions=data;if(mode==='interview')renderQuestions()}).catch(e=>toast(e.message));
        return true;
      }
      generation++;
      return false;
    }
  };
}
