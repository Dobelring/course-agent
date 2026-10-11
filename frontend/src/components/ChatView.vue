<script setup>
import { onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import {
  cancelRun,
  chatStream,
  deleteConversation,
  fetchConversation,
  fetchConversations,
  renameConversation,
  resolveApproval,
} from '../api'

const props = defineProps({
  token: { type: String, required: true },
  showHistory: { type: Boolean, default: false },
})

const messages = ref([])
const input = ref('')
const busy = ref(false)
const error = ref('')
const conversations = ref([])
const currentConvId = ref(null)
const currentRunId = ref(null) // 当前生成的 run_id，用于中途取消
// 输入草稿按会话隔离：key 为会话 id 字符串，'new' 表示「新对话」未发送状态。
// 切换会话时保存当前草稿并恢复目标草稿——输入的内容只属于它所在会话。
const drafts = ref({})

function draftKey() {
  return currentConvId.value == null ? 'new' : String(currentConvId.value)
}

function saveDraft() {
  drafts.value[draftKey()] = input.value
}

function restoreDraft() {
  input.value = drafts.value[draftKey()] ?? ''
}

function fromHistory(m) {
  if (m.role === 'user') return { role: 'user', content: m.content }
  return {
    role: 'assistant',
    content: m.content,
    tools: (m.tool_calls || []).map((t) => ({
      name: t.name,
      status: '完成',
    })),
    sources: m.retrieval_sources || null,
    plan: m.plan || null,
    latency_ms: m.latency_ms,
    first_token_ms: m.first_token_ms,
    usage: m.input_tokens != null
      ? { input_tokens: m.input_tokens, output_tokens: m.output_tokens, cost_cny: m.cost_cny }
      : null,
  }
}

// 工具名映射为中文，卡片只显示名称与状态，不展示原始 JSON
const TOOL_LABELS = {
  get_my_schedule: '查询课表',
  search_course_documents: '检索课程资料',
  list_my_documents: '查询课程资料列表',
  save_study_plan: '保存学习计划',
  get_upcoming_assignments: '查询近期作业',
  get_my_scores: '查询成绩',
}

const STEP_STATUS = {
  pending: { icon: '○', label: '等待', cls: 'pending' },
  running: { icon: '◉', label: '执行中', cls: 'running' },
  done: { icon: '✓', label: '完成', cls: 'done' },
  failed: { icon: '✗', label: '失败', cls: 'failed' },
  skipped: { icon: '⊘', label: '跳过', cls: 'skipped' },
}

function stepMeta(status) {
  return STEP_STATUS[status] || STEP_STATUS.pending
}

// 写操作审批：三态决策（允许 / 拒绝 / 填修改意见后执行）。
// 不阻塞读流：弹窗是独立状态，决策经独立接口回传（SSE 流继续读取）
const approvalBox = ref({ visible: false, ev: null, feedback: '' })

function handleApproval(ev, assistant) {
  assistant.notices.push(`⏳ 等待确认：是否执行「${ev.tool}」？（${ev.timeout_s} 秒内有效）`)
  approvalBox.value = { visible: true, ev, feedback: '' }
}

async function decideApproval(action) {
  const { ev, feedback } = approvalBox.value
  if (!ev) return
  approvalBox.value = { visible: false, ev: null, feedback: '' }
  try {
    await resolveApproval(ev.approval_id, action, feedback.trim(), props.token)
  } catch (e) {
    ElMessage.warning(e.message)
  }
}

// 助手回答按 Markdown 渲染（表格/加粗/列表），DOMPurify 防 XSS
function renderMd(text) {
  return DOMPurify.sanitize(marked.parse(text || '', { breaks: true }))
}

async function loadConversations() {
  if (!props.showHistory) return
  try {
    conversations.value = await fetchConversations(props.token)
  } catch {
    /* 忽略历史加载失败 */
  }
}

async function newConversation() {
  if (busy.value) return
  // 懒创建：不立即建会话——发送首条消息时后端才创建。
  // 「点了新对话但没发消息」不会在侧栏留下空白会话，切换走自然无残留。
  saveDraft()
  currentConvId.value = null
  messages.value = []
  restoreDraft()
}

async function openConversation(id) {
  if (busy.value) return
  saveDraft() // 当前会话的未发送草稿留存
  try {
    const conv = await fetchConversation(props.token, id)
    currentConvId.value = conv.id
    messages.value = conv.messages.map(fromHistory)
    restoreDraft() // 目标会话的草稿恢复（没有则输入框为空白）
  } catch (e) {
    ElMessage.error(e.message)
  }
}

async function removeConversation(id) {
  try {
    await deleteConversation(props.token, id)
    delete drafts.value[String(id)]
    if (currentConvId.value === id) {
      currentConvId.value = null
      messages.value = []
      restoreDraft()
    }
    await loadConversations()
  } catch (e) {
    ElMessage.error(e.message)
  }
}

// 重命名会话：弹输入框预填当前标题，成功后就地更新侧栏
async function renameConv(c) {
  try {
    const { value } = await ElMessageBox.prompt('输入新的对话名称', '重命名对话', {
      inputValue: c.title,
      confirmButtonText: '保存',
      cancelButtonText: '取消',
      inputValidator: (v) => (v && v.trim() ? true : '名称不能为空'),
    })
    const updated = await renameConversation(props.token, c.id, value.trim())
    c.title = updated.title
    ElMessage.success('已重命名')
  } catch {
    /* 用户取消或校验失败，忽略 */
  }
}

const SUGGESTIONS = [
  '我明天有什么课？',
  '我最近有什么作业要交？',
  '我的 GPA 多少，哪门最弱？',
  '帮我制定下周的学习计划',
]

// 运行模式：auto=自动路由 / direct=单循环 / plan=规划执行
const mode = ref('auto')
const MODE_OPTIONS = [
  { value: 'auto', label: '自动' },
  { value: 'direct', label: '直接回答' },
  { value: 'plan', label: '规划执行' },
]

function findStep(assistant, id) {
  return assistant.plan?.steps?.find((s) => s.id === id)
}

async function doSend(text, opts = {}) {
  if (!text || busy.value) return
  error.value = ''
  messages.value.push({ role: 'user', content: text })
  const assistant = ref({
    role: 'assistant',
    content: '',
    tools: [],
    plan: null,
    mode: null,
    notices: [],
  })
  messages.value.push(assistant.value)
  busy.value = true
  const runId = crypto.randomUUID() // 服务端注册，用于中途取消
  currentRunId.value = runId

  const updateTool = (name, patch) => {
    const t = assistant.value.tools.find((x) => x.name === name)
    if (t) Object.assign(t, patch)
  }

  try {
    await chatStream(text, props.token, (ev) => {
      switch (ev.type) {
        case 'mode':
          assistant.value.mode = ev.mode
          break
        case 'context':
          if (ev.compressed) {
            assistant.value.notices.push(`已将更早的 ${ev.summary_chars} 字历史压缩为摘要`)
          }
          break
        case 'approval_required':
          // 不阻塞读流：弹窗确认后异步回传决策，服务器会一直等（带超时）
          handleApproval(ev, assistant.value)
          break
        case 'approval_resolved':
          assistant.value.notices.push(
            ev.approved ? `✓ 已授权执行「${ev.tool}」` : `✗ 已拒绝「${ev.tool}」（${ev.reason}）`,
          )
          break
        case 'plan':
          assistant.value.plan = { steps: ev.steps, revised: 0, degraded: false }
          break
        case 'step_start': {
          const s = findStep(assistant.value, ev.id)
          if (s) s.status = 'running'
          break
        }
        case 'step_text': {
          // 步骤执行中的流式输出：实时挂到对应步骤卡片（保留尾部，防止无限增长）
          const s = findStep(assistant.value, ev.id)
          if (s) s.live = ((s.live || '') + ev.delta).slice(-400)
          break
        }
        case 'step_retry': {
          const s = findStep(assistant.value, ev.id)
          if (s) s.retries = (s.retries || 0) + 1
          break
        }
        case 'step_end': {
          const s = findStep(assistant.value, ev.id)
          if (s) {
            s.status = ev.status
            s.result = ev.result
            s.error = ev.error
          }
          break
        }
        case 'replan':
          if (assistant.value.plan) {
            assistant.value.plan.revised = ev.revised
            // 替换后续步骤（保留已执行的历史）
            const doneOrFailed = assistant.value.plan.steps.filter(
              (s) => s.status !== 'pending',
            )
            assistant.value.plan.steps = [...doneOrFailed, ...ev.steps]
            assistant.value.notices.push(`第 ${ev.revised} 次重规划：调整剩余步骤`)
          }
          break
        case 'degraded':
          assistant.value.notices.push(`⚠ ${ev.reason}`)
          break
        case 'failover':
          assistant.value.notices.push(`⚠ 模型调用失败，已切换备用路由：${ev.from} → ${ev.to}`)
          break
        case 'reflection':
          assistant.value.notices.push(
            `🔍 质量自检 ${ev.quality}/5${ev.issues?.length ? '：' + ev.issues.join('；') : ''}`
          )
          if (ev.revised) assistant.value.content = ev.revised
          break
        case 'fallback':
          assistant.value.notices.push(`↩ 已降级为直接回答：${ev.reason}`)
          break
        case 'tool_start':
          assistant.value.tools.push({ name: ev.name, status: '运行中' })
          break
        case 'tool_end':
          updateTool(ev.name, { status: '完成' })
          break
        case 'text':
          assistant.value.content += ev.delta
          break
        case 'error':
          error.value = ev.message
          break
        case 'cancelled':
          assistant.value.notices.push(`⏹ 已停止生成（已保留 ${ev.partial_chars} 字内容）`)
          // 计划执行到一半被停：plan 卡片标记暂停，展示「继续执行」入口
          if (ev.plan_run_id && assistant.value.plan) {
            assistant.value.plan.paused = true
            assistant.value.plan.plan_run_id = ev.plan_run_id
          }
          break
        case 'done':
          assistant.value.latency_ms = ev.latency_ms
          assistant.value.first_token_ms = ev.first_token_ms
          if (ev.plan) assistant.value.plan = ev.plan
          if (ev.usage) assistant.value.usage = ev.usage
          if (ev.conversation_id && currentConvId.value !== ev.conversation_id) {
            currentConvId.value = ev.conversation_id
            if (props.showHistory) loadConversations()
          }
          break
      }
    }, currentConvId.value, mode.value, runId, opts.resumePlanId ?? null)
  } catch (e) {
    error.value = e.message
  } finally {
    busy.value = false
    currentRunId.value = null
    if (props.showHistory) loadConversations()
  }
}

// 中途停止：请求服务端取消后台任务，SSE 流随后收到 cancelled 事件并收尾
async function stop() {
  if (!busy.value || !currentRunId.value) return
  try {
    await cancelRun(currentRunId.value, props.token)
  } catch (e) {
    // 任务已结束等竞态：忽略，等流自行收尾
  }
}

// 人工接管：续跑暂停的计划——done 步骤复用产出，只补 pending 部分
function resumePlan(planRunId) {
  if (busy.value) return
  doSend('继续执行暂停的计划', { resumePlanId: planRunId })
}

function send() {
  const text = input.value.trim()
  if (!text || busy.value) return
  input.value = ''
  delete drafts.value[draftKey()] // 草稿已随消息发出，清除
  doSend(text)
}

onMounted(() => {
  if (props.showHistory) loadConversations()
})
</script>

<template>
  <div class="chat" :class="{ withHistory: showHistory }">
    <!-- 会话历史侧栏 -->
    <aside v-if="showHistory" class="history">
      <el-button type="primary" class="new-btn" :disabled="busy" @click="newConversation">
        + 新对话
      </el-button>
      <div
        v-for="c in conversations"
        :key="c.id"
        class="conv"
        :class="{ active: currentConvId === c.id }"
        @click="openConversation(c.id)"
      >
        <div class="conv-title">{{ c.title }}</div>
        <div class="conv-meta">{{ c.msg_count }} 条回复</div>
        <el-button
          class="conv-ren"
          size="small"
          text
          title="重命名"
          @click.stop="renameConv(c)"
        >
          ✎
        </el-button>
        <el-button
          class="conv-del"
          size="small"
          text
          type="danger"
          @click.stop="removeConversation(c.id)"
        >
          删除
        </el-button>
      </div>
      <p v-if="!conversations.length" class="history-empty">暂无历史对话</p>
    </aside>

    <div class="main">
      <div class="list">
        <div v-if="!messages.length" class="empty">
          <p class="hello">你好，我是校园课程智能助手</p>
          <p class="sub">可以帮你查课表、盯作业、背卡片、答疑资料、做计划——点一个问题直接开始，或在下方输入</p>
          <div class="suggestions">
            <button v-for="s in SUGGESTIONS" :key="s" class="chip" @click="doSend(s)">
              {{ s }}
            </button>
          </div>
        </div>

        <div v-for="(m, i) in messages" :key="i" class="msg" :class="m.role">
          <div class="bubble">
            <template v-if="m.role === 'user'">{{ m.content }}</template>
            <template v-else>
              <!-- 运行模式徽标 -->
              <div v-if="m.mode" class="mode-badge">
                {{ m.mode === 'plan' ? '📋 规划执行' : '💬 直接回答' }}
              </div>
              <!-- 规划步骤卡片 -->
              <div v-if="m.plan && m.plan.steps && m.plan.steps.length" class="plan">
                <div class="plan-head">
                  执行计划
                  <span v-if="m.plan.revised" class="plan-revised">已重规划 {{ m.plan.revised }} 次</span>
                  <button
                    v-if="m.plan.paused && m.plan.plan_run_id"
                    class="plan-resume"
                    title="从暂停处继续执行（已完成的步骤不重跑）"
                    @click="resumePlan(m.plan.plan_run_id)"
                  >
                    ▶ 继续执行
                  </button>
                </div>
                <div
                  v-for="s in m.plan.steps"
                  :key="s.id"
                  class="plan-step"
                  :class="stepMeta(s.status).cls"
                >
                  <span class="step-icon">{{ stepMeta(s.status).icon }}</span>
                  <span class="step-title">{{ s.title }}</span>
                  <span class="step-status">{{ stepMeta(s.status).label }}</span>
                  <span v-if="s.retries" class="step-retries">重试 {{ s.retries }} 次</span>
                  <!-- 步骤执行中的流式输出（实时吐字） -->
                  <span v-if="s.live" class="step-live">{{ s.live }}</span>
                </div>
              </div>
              <!-- 降级/压缩等过程提示 -->
              <div v-for="(n, ni) in m.notices" :key="'n' + ni" class="notice">{{ n }}</div>
              <!-- 工具调用过程卡片（只显示中文名称与状态） -->
              <div v-for="(t, ti) in m.tools" :key="ti" class="tool">
                <div class="tool-head">
                  <span class="tool-name">🔧 {{ TOOL_LABELS[t.name] || t.name }}</span>
                  <span class="tool-status" :class="t.status === '完成' ? 'ok' : 'run'">{{ t.status }}</span>
                </div>
              </div>
              <!-- 检索溯源 -->
              <div v-if="m.sources && m.sources.length" class="sources">
                <div class="sources-title">📚 检索来源</div>
                <div v-for="(s, si) in m.sources" :key="si" class="source">
                  <span class="source-file">{{ s.filename }}</span>
                  <span class="source-score">相似度 {{ s.score }}</span>
                </div>
              </div>
              <div v-if="m.content" class="answer md" v-html="renderMd(m.content)"></div>
              <div v-else-if="m.tools.length || (m.plan && m.plan.steps)" class="thinking">正在组织回答…</div>
              <div v-if="m.latency_ms" class="latency">
                <template v-if="m.first_token_ms">首字 {{ (m.first_token_ms / 1000).toFixed(1) }} 秒 · </template>总耗时 {{ (m.latency_ms / 1000).toFixed(1) }} 秒<template v-if="m.usage && m.usage.total_tokens"> · {{ m.usage.total_tokens }} tokens · ¥{{ (m.usage.cost_cny || 0).toFixed(4) }}</template>
              </div>
            </template>
          </div>
        </div>
      </div>

      <div class="composer">
        <p v-if="error" class="error">{{ error }}</p>
        <div class="row">
          <select v-model="mode" class="mode-select" :disabled="busy" title="运行模式">
            <option v-for="o in MODE_OPTIONS" :key="o.value" :value="o.value">{{ o.label }}</option>
          </select>
          <input
            v-model="input"
            placeholder="输入你的问题…"
            :disabled="busy"
            @keyup.enter="send"
          />
          <button
            v-if="busy"
            class="stop-btn"
            title="停止本次生成"
            @click="stop"
          >
            ■ 停止
          </button>
          <button v-else :disabled="!input.trim()" @click="send">
            发送
          </button>
        </div>
      </div>
    </div>

    <!-- 写操作审批：允许 / 拒绝 / 填修改意见后执行（三态） -->
    <el-dialog
      v-model="approvalBox.visible"
      title="需要你的授权"
      width="520px"
      :close-on-click-modal="false"
    >
      <p class="approval-line">工具「{{ approvalBox.ev?.tool }}」将写入数据：</p>
      <pre class="approval-args">{{ approvalBox.ev?.args }}</pre>
      <el-input
        v-model="approvalBox.feedback"
        type="textarea"
        :rows="2"
        placeholder="修改意见（选「修改后执行」时必填；允许/拒绝可不填）"
      />
      <template #footer>
        <el-button type="danger" plain @click="decideApproval('reject')">拒绝</el-button>
        <el-button
          type="warning"
          :disabled="!approvalBox.feedback.trim()"
          @click="decideApproval('revise')"
        >
          修改后执行
        </el-button>
        <el-button type="primary" @click="decideApproval('approve')">允许执行</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.chat {
  display: flex;
  height: calc(100vh - 140px);
  min-height: 460px;
  gap: 14px;
}

.chat:not(.withHistory) {
  display: block;
  height: calc(100vh - 140px);
  min-height: 420px;
}

.history {
  width: 230px;
  flex-shrink: 0;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  padding: 10px;
  overflow-y: auto;
  background: #fff;
}

.new-btn {
  width: 100%;
  margin-bottom: 10px;
}

.conv {
  position: relative;
  padding: 8px 10px;
  border-radius: 8px;
  cursor: pointer;
  margin-bottom: 6px;
  border: 1px solid transparent;
}

.conv:hover {
  background: #f3f4f6;
}

.conv.active {
  background: #ecf5ff;
  border-color: #409eff;
}

.conv-title {
  font-size: 13px;
  font-weight: 500;
  color: #1f2937;
  padding-right: 70px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.conv-meta {
  font-size: 11px;
  color: #9ca3af;
  margin-top: 2px;
}

/* 重命名在删除左侧，两个按钮都停靠右上 */
.conv-ren {
  position: absolute;
  right: 44px;
  top: 2px;
}

.conv-del {
  position: absolute;
  right: 2px;
  top: 2px;
}

.history-empty {
  color: #9ca3af;
  font-size: 12px;
  text-align: center;
  margin-top: 20px;
}

.main {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.list {
  flex: 1;
  overflow-y: auto;
  padding: 12px 4px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.empty {
  margin: auto;
  color: #6b7280;
  font-size: 14px;
  text-align: center;
}

.empty .hello {
  font-size: 18px;
  font-weight: 600;
  color: #111827;
  margin-bottom: 6px;
}

.empty .sub {
  margin-bottom: 16px;
}

.suggestions {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 10px;
  max-width: 480px;
  margin: 0 auto;
}

.suggestions .chip {
  padding: 8px 16px;
  border: 1px solid #91caff;
  border-radius: 999px;
  background: #fff;
  color: #1890ff;
  font-size: 13px;
  cursor: pointer;
  transition: background 0.15s, box-shadow 0.15s;
}

.suggestions .chip:hover {
  background: #e6f7ff;
  box-shadow: 0 1px 6px rgba(24, 144, 255, 0.2);
}

.msg {
  display: flex;
}

.msg.user {
  justify-content: flex-end;
}

.bubble {
  max-width: 78%;
  padding: 10px 14px;
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
}

.msg.user .bubble {
  background: #409eff;
  color: #fff;
}

.msg.assistant .bubble {
  background: #fff;
  border: 1px solid #e5e7eb;
  width: 78%;
}

.tool {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 8px 10px;
  margin-bottom: 8px;
  font-size: 12px;
}

.mode-badge {
  display: inline-block;
  font-size: 11px;
  color: #6366f1;
  background: #eef2ff;
  border: 1px solid #c7d2fe;
  border-radius: 999px;
  padding: 2px 10px;
  margin-bottom: 8px;
}

.plan {
  background: #f5f3ff;
  border: 1px solid #ddd6fe;
  border-radius: 8px;
  padding: 8px 10px;
  margin-bottom: 8px;
  font-size: 12px;
}

.plan-head {
  font-weight: 600;
  color: #5b21b6;
  margin-bottom: 6px;
}

.plan-revised {
  font-weight: 400;
  color: #b45309;
  margin-left: 8px;
  font-size: 11px;
}

/* 暂停计划的续跑按钮 */
.plan-resume {
  margin-left: auto;
  padding: 2px 10px;
  border: 1px solid #16a34a;
  border-radius: 6px;
  background: #f0fdf4;
  color: #16a34a;
  font-size: 11px;
  cursor: pointer;
}

.plan-resume:hover {
  background: #dcfce7;
}

.plan-step {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  padding: 3px 0;
  color: #4c1d95;
}

.plan-step .step-icon {
  width: 14px;
  text-align: center;
}

.plan-step.done .step-icon { color: #16a34a; }
.plan-step.running .step-icon { color: #2563eb; animation: pulse 1s infinite; }
.plan-step.failed .step-icon { color: #dc2626; }
.plan-step.skipped .step-icon { color: #9ca3af; }
.plan-step.pending { color: #8b7fc7; }

.step-title { flex: 1; }
.step-status { font-size: 11px; opacity: 0.75; }
.step-retries { font-size: 11px; color: #b45309; }

/* 步骤执行中的流式输出：占满整行、灰色小字、超长隐藏 */
.step-live {
  flex-basis: 100%;
  font-size: 11px;
  color: #6b7280;
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 3.6em;
  overflow: hidden;
}

@keyframes pulse {
  50% { opacity: 0.35; }
}

.notice {
  font-size: 12px;
  color: #92400e;
  background: #fffbeb;
  border: 1px dashed #fde68a;
  border-radius: 8px;
  padding: 5px 10px;
  margin-bottom: 6px;
}

.mode-select {
  flex-shrink: 0;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 0 8px;
  height: 40px;
  background: #fff;
  font-size: 13px;
  color: #374151;
  outline: none;
}

.mode-select:disabled { opacity: 0.5; }

.tool-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.tool-name {
  font-weight: 600;
  color: #0f172a;
}

.tool-status.ok {
  color: #16a34a;
}

.tool-status.run {
  color: #2563eb;
}

.sources {
  background: #fffbeb;
  border: 1px solid #fde68a;
  border-radius: 8px;
  padding: 8px 10px;
  margin-bottom: 8px;
  font-size: 12px;
}

.sources-title {
  font-weight: 600;
  color: #92400e;
  margin-bottom: 4px;
}

.source {
  display: flex;
  justify-content: space-between;
  color: #78350f;
}

.source-score {
  color: #b45309;
}

.answer {
  margin-top: 2px;
}

/* Markdown 渲染样式：覆盖气泡的 pre-wrap，统一表格/列表观感 */
.md {
  white-space: normal;
  line-height: 1.7;
}

.md :deep(p) {
  margin: 6px 0;
}

.md :deep(p:first-child) {
  margin-top: 0;
}

.md :deep(p:last-child) {
  margin-bottom: 0;
}

.md :deep(table) {
  border-collapse: collapse;
  margin: 8px 0;
  font-size: 13px;
  display: block;
  overflow-x: auto;
}

.md :deep(th),
.md :deep(td) {
  border: 1px solid #e5e7eb;
  padding: 6px 10px;
  text-align: left;
}

.md :deep(th) {
  background: #f9fafb;
  color: #4b5563;
  font-weight: 600;
}

.md :deep(ul),
.md :deep(ol) {
  padding-left: 20px;
  margin: 6px 0;
}

.md :deep(li) {
  margin: 2px 0;
}

.md :deep(strong) {
  color: #0f172a;
}

.md :deep(code) {
  background: #f1f5f9;
  border-radius: 4px;
  padding: 1px 5px;
  font-size: 12px;
}

.thinking {
  color: #9ca3af;
  font-size: 13px;
}

.latency {
  color: #9ca3af;
  font-size: 11px;
  margin-top: 6px;
  text-align: right;
}

.composer {
  border-top: 1px solid #e5e7eb;
  padding-top: 12px;
}

.row {
  display: flex;
  gap: 8px;
}

.row input {
  flex: 1;
  padding: 10px 12px;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  font-size: 14px;
  outline: none;
}

.row button {
  padding: 0 20px;
  border: none;
  border-radius: 8px;
  background: #409eff;
  color: #fff;
  font-size: 14px;
  cursor: pointer;
}

.row button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.row button.stop-btn {
  background: #dc2626;
}

/* 审批弹窗内容 */
.approval-line {
  margin: 0 0 6px;
  font-size: 14px;
  color: #1f2937;
}

.approval-args {
  margin: 0 0 10px;
  padding: 8px 10px;
  max-height: 140px;
  overflow-y: auto;
  background: #f9fafb;
  border-radius: 6px;
  font-size: 12px;
  color: #374151;
  white-space: pre-wrap;
  word-break: break-all;
}

.error {
  color: #dc2626;
  font-size: 13px;
  margin-bottom: 8px;
}
</style>
