import katex from 'katex';
import {pythonLanguage} from '@codemirror/lang-python';
import {highlightTree, classHighlighter} from '@lezer/highlight';

const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));

export function codeBlock(code) {
  let cursor = 0;
  let html = '';
  highlightTree(pythonLanguage.parser.parse(code), classHighlighter, (from, to, classes) => {
    html += escape(code.slice(cursor, from));
    html += `<span class="${classes}">${escape(code.slice(from, to))}</span>`;
    cursor = to;
  });
  return `<pre class="python-code"><code>${html + escape(code.slice(cursor))}</code></pre>`;
}

function inputRows(problem) {
  const defaults = Object.fromEntries(problem.signature.split(',').map(part => part.trim().split('=')));
  return problem.inputs.map(input => `<tr>
    <td><code>${escape(input.name)}</code>${defaults[input.name] === undefined ? '' : `<span class="param-default">默认 ${escape(defaults[input.name])}</span>`}</td>
    <td><span class="io-type">${escape(input.type)}</span><code class="io-shape">${escape(input.shape)}</code></td>
    <td>${escape(input.description)}</td>
  </tr>`).join('');
}

export function problemView(p, badge) {
  const tuple = p.returns.length > 1;
  const signature = `def solve(${p.signature}):`;
  const resultSignature = tuple ? `return (${p.returns.map(r => r.name).join(', ')})` : `return ${p.returns[0].name}`;
  const rules = p.statement.split(/\n|。/).map(s => s.trim()).filter(Boolean);
  const formula = p.formulas.length ? `<section id="problem-formula"><h3><span>03</span> 计算公式</h3><div class="formula-group">${p.formulas.map(f => `<div class="math-equation">${katex.renderToString(f,{displayMode:true,output:'mathml',throwOnError:true})}</div>`).join('')}</div>${['ppo_loss','gspo_loss','dapo_loss','grpo_loss'].includes(p.id)?'<p class="formula-key">ℓ 为 log probability，m 为有效位置掩码；仅对有效 token 求和。</p>':''}</section>` : '';
  return `<div class="problem-heading"><div class="problem-meta">${badge(p.difficulty)}<span>${escape(p.category)}</span></div><h2>${escape(p.title)}</h2></div>
    <nav class="doc-outline" aria-label="题目章节"><a href="#problem-inputs">输入参数</a><a href="#problem-returns">返回要求</a><a href="#problem-rules">计算规则</a><button data-open-cases>测试示例 ↗</button></nav>
    <div class="signature-block"><div class="block-label">PYTHON 接口</div>${codeBlock(signature)}<div class="return-preview"><span>返回</span><code>${escape(tuple?'tuple':p.returns[0].type)}</code><span>${escape(tuple?p.returns.map(r=>r.name).join(' → '):p.returns[0].shape)}</span></div></div>
    <section id="problem-inputs"><h3><span>01</span> 输入参数</h3><div class="io-table-wrap"><table class="io-table"><thead><tr><th>参数</th><th>类型 / Shape</th><th>含义与要求</th></tr></thead><tbody>${inputRows(p)}</tbody></table></div><p class="shape-note">维度字母表示各题中的大小，见参数描述；<code>[]</code> 表示零维标量，<code>...</code> 表示任意前导维度。浮点输入在测试中使用兼容的 dtype 和 device。</p></section>
    <section id="problem-returns"><h3><span>02</span> 返回要求</h3><div class="return-contract">${codeBlock(resultSignature)}${tuple?'<p class="return-order">必须按以下顺序返回一个 Python 元组。</p>':''}<ol class="return-items">${p.returns.map(r=>`<li><div><code>${escape(r.name)}</code><span class="io-type">${escape(r.type)}</span><code class="io-shape">${escape(r.shape)}</code></div><p>${escape(r.description)}</p></li>`).join('')}</ol></div></section>
    ${formula}<section id="problem-rules"><h3><span>${p.formulas.length?'04':'03'}</span> 计算规则与边界</h3><ul class="rule-list">${rules.map(rule=>`<li>${escape(rule)}。</li>`).join('')}</ul></section>
    <section><h3><span>✓</span> 验证方式</h3><div class="verification-note"><strong>${p.cases.length} 种场景 · ${p.cases.length*3} 次执行</strong><p>${p.gradient?'比较前向结果，以及输入和参数的梯度。':'比较返回值的结构、形状、dtype 和数值。'}每种场景使用 17、41、103 三个种子。</p><p>“运行示例”只运行第一个场景。“提交测试”运行全部场景。算法/API/复杂度限制仍需自行检查。</p></div></section>
    <details class="hint-box"><summary>需要一点提示？</summary><p>${escape(p.hint || '从输入 shape 推导每一步输出 shape，再检查求和维度和边界。先用小样本验证，再处理 batch 和梯度。')}</p></details>
    <div class="solution-footer"><button id="show-answer">查看参考实现</button><span>建议先独立完成</span></div>
    ${p.sources.length?`<div class="source-links">${p.sources.map((url,i)=>`<a href="${escape(url)}" target="_blank" rel="noreferrer">原始资料 ${i+1} ↗</a>`).join('')}</div>`:''}
    <details class="runtime-details"><summary>执行环境与调试说明</summary><p>代码在本机 Python / CPU PyTorch 子进程执行，限时 30 秒，只用于你自己的代码，不是安全沙箱。快照在指定行执行前记录，最多40条。</p></details>`;
}

export function casesView(p) {
  return `<div class="problem-heading"><p class="section-kicker">EXAMPLES & TESTS</p><h2>测试用例</h2><p class="problem-lead">先阅读每个场景的命名参数，再运行或载入调试。下面的期望输出由参考实现实际计算。</p></div>${p.cases.map((c,i)=>`<article class="example-card"><div class="example-heading"><span>示例 ${String(i+1).padStart(2,'0')}</span><h3>${escape(c.name)}</h3></div><div class="block-label">输入 · seed 17</div>${codeBlock(Object.entries(c.arguments).map(([name,value])=>name+' = '+value).join('\n\n'))}<details class="fixture-details"><summary>查看可运行的输入构造代码</summary>${codeBlock('torch.manual_seed(17)\n'+c.setup.replace(/; /g,';\n'))}</details><div class="block-label">调用方式</div>${codeBlock('result = solve('+Object.keys(c.arguments).join(', ')+')\nprint(result)')}<div class="block-label">期望返回值</div>${codeBlock(c.expected)}<p class="shape-note">数值比较使用容差；grad_fn 名称取决于实现，不需要一致。</p><button data-debug-case="${i}">载入自定义调试 →</button></article>`).join('')}`;
}
