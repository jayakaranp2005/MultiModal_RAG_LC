<script>
	import { onMount } from 'svelte';
	import { health, getIndexed, uploadPdf, ask } from './lib/api';

	const MAX_FILE_SIZE_MB = 25;
	const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;
	const SOURCE_PREVIEW_LIMIT = 600;

	let healthLoading = true;
	let healthError = '';
	let healthData = null;

	let indexedLoading = false;
	let indexedError = '';
	let indexedPdfs = [];

	let selectedFile = null;
	let reindex = false;
	let uploadState = 'idle';
	let uploadError = '';
	let uploadResult = null;

	let question = '';
	let askLoading = false;
	let askError = '';
	let answerData = null;
	let showSources = false;
	let copyState = 'idle';
	let chatMessages = [];

	$: backendReady = Boolean(healthData && healthData.ready === true);
	$: statusLabel = healthLoading ? 'Checking...' : backendReady ? 'Ready' : 'Not ready';
	$: uploadBusy = uploadState === 'uploading' || uploadState === 'indexing';

	function truncateSource(source) {
		const text = typeof source === 'string' ? source : String(source ?? '');
		return text.length > SOURCE_PREVIEW_LIMIT
			? `${text.slice(0, SOURCE_PREVIEW_LIMIT)}...`
			: text;
	}

	async function loadHealth() {
		healthLoading = true;
		healthError = '';

		try {
			healthData = await health();
		} catch (error) {
			healthError = error.message;
			healthData = null;
		} finally {
			healthLoading = false;
		}
	}

	async function loadIndexed() {
		indexedLoading = true;
		indexedError = '';

		try {
			const data = await getIndexed();
			indexedPdfs = Array.isArray(data.indexed) ? data.indexed : [];
		} catch (error) {
			indexedError = error.message;
			indexedPdfs = [];
		} finally {
			indexedLoading = false;
		}
	}

	function onFileChange(event) {
		uploadError = '';
		uploadResult = null;
		const [file] = event.currentTarget.files || [];
		selectedFile = file || null;
	}

	function validatePdfFile(file) {
		if (!file) {
			return 'Please choose a PDF file before uploading.';
		}

		const hasPdfMimeType = file.type === 'application/pdf';
		const hasPdfFileName = file.name.toLowerCase().endsWith('.pdf');

		if (!hasPdfMimeType && !hasPdfFileName) {
			return 'Only PDF files are allowed.';
		}

		if (file.size > MAX_FILE_SIZE_BYTES) {
			return `File is too large. Maximum size is ${MAX_FILE_SIZE_MB} MB.`;
		}

		return '';
	}

	async function handleUpload() {
		uploadError = '';
		uploadResult = null;

		const validationMessage = validatePdfFile(selectedFile);
		if (validationMessage) {
			uploadState = 'error';
			uploadError = validationMessage;
			return;
		}

		uploadState = 'uploading';

		// Shift to "indexing" while waiting so users can tell the process is still active.
		const indexingHintTimer = setTimeout(() => {
			if (uploadState === 'uploading') {
				uploadState = 'indexing';
			}
		}, 900);

		try {
			uploadResult = await uploadPdf(selectedFile, reindex);
			uploadState = 'done';
			await loadIndexed();
		} catch (error) {
			uploadState = 'error';
			uploadError = error.message;
		} finally {
			clearTimeout(indexingHintTimer);
		}
	}

	async function handleAsk() {
		askError = '';
		copyState = 'idle';

		const trimmedQuestion = question.trim();
		if (!trimmedQuestion) {
			askError = 'Please enter a question before sending.';
			return;
		}

		chatMessages = [
			...chatMessages,
			{
				role: 'outgoing',
				text: trimmedQuestion,
				timestamp: Date.now()
			}
		];

		askLoading = true;

		try {
			const response = await ask(trimmedQuestion);
			answerData = {
				answer: String(response.answer || ''),
				sources: Array.isArray(response.sources) ? response.sources : [],
				image_count: Number(response.image_count || 0)
			};
			showSources = false;
			chatMessages = [
				...chatMessages,
				{
					role: 'incoming',
					text: answerData.answer,
					image_count: answerData.image_count,
					sources: answerData.sources,
					timestamp: Date.now()
				}
			];
			question = '';
		} catch (error) {
			askError = error.message;
		} finally {
			askLoading = false;
		}
	}

	function onQuestionKeydown(event) {
		if (event.key === 'Enter' && event.ctrlKey) {
			event.preventDefault();
			handleAsk();
		}
	}

	async function copyAnswer() {
		if (!answerData || !answerData.answer) {
			return;
		}

		try {
			await navigator.clipboard.writeText(answerData.answer);
			copyState = 'copied';
		} catch (_error) {
			copyState = 'failed';
		}
	}

	onMount(async () => {
		await Promise.all([loadHealth(), loadIndexed()]);
	});
</script>

<main class="app-shell">
	<header class="topbar card">
		<div>
			<h1>MultiModal RAG</h1>
			<p class="subtitle">Upload, index, and chat with your PDF knowledge base</p>
		</div>
		<div class="topbar-right">
			<span class:ready={backendReady} class:danger={!backendReady} class="status-pill">{statusLabel}</span>
		</div>
	</header>

	{#if healthError}
		<p class="error-banner">Health check failed: {healthError}</p>
	{/if}

	<section class="panel-grid">
		<article class="card panel">
			<h2>PDF Indexing</h2>

			<label for="pdf-file">PDF file</label>
			<input id="pdf-file" type="file" accept="application/pdf" on:change={onFileChange} disabled={uploadBusy} />

			<label class="checkbox-row" for="reindex">
				<input id="reindex" type="checkbox" bind:checked={reindex} disabled={uploadBusy} />
				<span>Reindex if already indexed</span>
			</label>

			<button class="primary" type="button" on:click={handleUpload} disabled={uploadBusy}>
				{#if uploadBusy}
					Working...
				{:else}
					Upload &amp; Index
				{/if}
			</button>

			<p class="state-label">Upload state: <strong>{uploadState}</strong></p>

			{#if uploadError}
				<p class="error-banner">{uploadError}</p>
			{/if}

			{#if uploadResult}
				<div class="result-card">
					<h3>Last Upload</h3>
					<p><strong>Filename:</strong> {uploadResult.filename}</p>
					<p><strong>Status:</strong> {uploadResult.status}</p>
					<p><strong>Texts:</strong> {uploadResult.texts}</p>
					<p><strong>Tables:</strong> {uploadResult.tables}</p>
					<p><strong>Images:</strong> {uploadResult.images}</p>
				</div>
			{/if}

			<div class="section-head">
				<h3>Indexed PDFs</h3>
				<button type="button" on:click={loadIndexed} disabled={indexedLoading}>Refresh</button>
			</div>

			{#if indexedError}
				<p class="error-banner">{indexedError}</p>
			{/if}

			{#if indexedLoading}
				<p class="muted">Loading indexed files...</p>
			{:else if indexedPdfs.length === 0}
				<p class="muted">No indexed PDFs yet.</p>
			{:else}
				<ul class="indexed-list">
					{#each indexedPdfs as item}
						<li>{item}</li>
					{/each}
				</ul>
			{/if}
		</article>

		<article class="card panel">
			<h2>Chat</h2>
			<p class="muted">Ask questions and view responses as a conversation.</p>

			<section class="chat-thread" aria-live="polite">
				{#if chatMessages.length === 0}
					<p class="muted">No messages yet. Ask your first question.</p>
				{/if}

				{#each chatMessages as msg}
					<div class:incoming={msg.role === 'incoming'} class:outgoing={msg.role === 'outgoing'} class="chat-row">
						<div class="chat-bubble">
							<p class="chat-text">{msg.text}</p>
							{#if msg.role === 'incoming'}
								<p class="chat-meta"><strong>Image count:</strong> {msg.image_count || 0}</p>
							{/if}
						</div>
					</div>
				{/each}

				{#if askLoading}
					<div class="chat-row incoming">
						<div class="chat-bubble typing-bubble" role="status">
							<span class="dot"></span>
							<span class="dot"></span>
							<span class="dot"></span>
						</div>
					</div>
				{/if}
			</section>

			<label for="question">Question</label>
			<textarea
				id="question"
				rows="5"
				bind:value={question}
				on:keydown={onQuestionKeydown}
				placeholder="Ask a question about your indexed PDFs..."
				disabled={askLoading}
			/>

			<div class="chat-actions">
				<button class="primary" type="button" on:click={handleAsk} disabled={askLoading}>Ask</button>
				<span class="hint">Press Ctrl+Enter to send</span>
			</div>

			{#if askError}
				<p class="error-banner">{askError}</p>
			{/if}

			{#if answerData}
				<div class="result-card">
					<div class="section-head compact">
						<h3>Latest response details</h3>
						<button type="button" class="ghost" on:click={copyAnswer}>Copy answer</button>
					</div>
					<p class="answer-text">{answerData.answer}</p>
					<p><strong>Image count:</strong> {answerData.image_count}</p>

					{#if copyState === 'copied'}
						<p class="hint">Answer copied to clipboard.</p>
					{:else if copyState === 'failed'}
						<p class="error-banner">Clipboard copy failed. Copy manually from the answer text.</p>
					{/if}

					{#if answerData.sources.length > 0}
						<button type="button" class="sources-toggle" on:click={() => (showSources = !showSources)}>
							{showSources ? 'Hide Sources' : 'Show Sources'}
						</button>

						{#if showSources}
							<ol class="sources-list">
								{#each answerData.sources as source}
									<li>{truncateSource(source)}</li>
								{/each}
							</ol>
						{/if}
					{/if}
				</div>
			{/if}
		</article>
	</section>
</main>

<style>
	:global(body) {
		margin: 0;
		background: linear-gradient(180deg, #f4f4f5 0%, #eceef2 100%);
		font-family: 'SF Pro Display', 'Avenir Next', 'Helvetica Neue', sans-serif;
		color: #1f2a37;
	}

	:global(*) {
		box-sizing: border-box;
	}

	.app-shell {
		max-width: 1160px;
		margin: 0 auto;
		padding: 1.2rem;
		display: grid;
		gap: 1rem;
	}

	.card {
		background: rgba(255, 255, 255, 0.86);
		backdrop-filter: blur(18px);
		border: 1px solid #e5e7eb;
		border-radius: 18px;
		padding: 1rem;
		box-shadow: 0 10px 34px rgba(15, 23, 42, 0.08);
	}

	.topbar {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		gap: 1rem;
	}

	h1,
	h2,
	h3 {
		margin: 0;
		font-weight: 600;
		letter-spacing: 0.01em;
	}

	.subtitle {
		margin: 0.35rem 0 0;
		color: #64748b;
	}

	.topbar-right {
		display: grid;
		gap: 0.4rem;
		justify-items: end;
	}

	.status-pill {
		display: inline-block;
		padding: 0.25rem 0.8rem;
		border-radius: 999px;
		font-size: 0.88rem;
		font-weight: 600;
		border: 1px solid transparent;
	}

	.status-pill.ready {
		background: #e8f8ef;
		color: #196c43;
		border-color: #bce8d1;
	}

	.status-pill.danger {
		background: #fdeeee;
		color: #8f1f1f;
		border-color: #f6caca;
	}

	.panel-grid {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 1rem;
	}

	.panel {
		display: grid;
		gap: 0.8rem;
		align-content: start;
	}

	label {
		font-size: 0.93rem;
		color: #334155;
	}

	input,
	textarea,
	button {
		font: inherit;
		border-radius: 12px;
		border: 1px solid #d1d5db;
		padding: 0.6rem 0.75rem;
	}

	textarea {
		resize: vertical;
		min-height: 150px;
	}

	button {
		cursor: pointer;
		background: #f3f4f6;
		color: #111827;
		transition: background-color 0.18s ease, transform 0.15s ease;
	}

	button:hover:enabled {
		background: #e5e7eb;
		transform: translateY(-1px);
	}

	button:disabled {
		opacity: 0.65;
		cursor: not-allowed;
	}

	button.primary {
		background: linear-gradient(135deg, #0f172a, #334155);
		border-color: #0f172a;
		color: #ffffff;
		font-weight: 600;
	}

	button.ghost {
		background: #ffffff;
		border-color: #e5e7eb;
	}

	.checkbox-row {
		display: flex;
		align-items: center;
		gap: 0.6rem;
	}

	.checkbox-row input {
		margin: 0;
	}

	.state-label,
	.hint,
	.muted {
		margin: 0;
		font-size: 0.9rem;
		color: #475569;
	}

	.error-banner {
		margin: 0;
		padding: 0.55rem 0.7rem;
		border-radius: 9px;
		background: #fff0f0;
		border: 1px solid #fecaca;
		color: #9f1239;
		font-size: 0.92rem;
	}

	.result-card {
		border: 1px solid #e2e8f0;
		border-radius: 12px;
		padding: 0.85rem;
		background: #f8fafc;
		display: grid;
		gap: 0.45rem;
	}

	.section-head.compact {
		align-items: center;
	}

	.result-card p {
		margin: 0;
	}

	.section-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.7rem;
	}

	.indexed-list,
	.sources-list {
		margin: 0;
		padding-left: 1.2rem;
		display: grid;
		gap: 0.35rem;
	}

	.indexed-list li,
	.sources-list li {
		line-height: 1.45;
		word-break: break-word;
	}

	.chat-actions {
		display: flex;
		align-items: center;
		gap: 0.65rem;
	}

	.chat-thread {
		display: grid;
		gap: 0.6rem;
		max-height: 360px;
		overflow-y: auto;
		padding: 0.7rem;
		border: 1px solid #e5e7eb;
		border-radius: 14px;
		background: #f8fafc;
	}

	.chat-row {
		display: flex;
	}

	.chat-row.outgoing {
		justify-content: flex-end;
	}

	.chat-row.incoming {
		justify-content: flex-start;
	}

	.chat-bubble {
		max-width: 85%;
		padding: 0.65rem 0.8rem;
		border-radius: 15px;
		word-break: break-word;
	}

	.chat-row.outgoing .chat-bubble {
		background: #dbeafe;
		color: #0f172a;
		border-bottom-right-radius: 6px;
	}

	.chat-row.incoming .chat-bubble {
		background: #ffffff;
		color: #111827;
		border: 1px solid #e5e7eb;
		border-bottom-left-radius: 6px;
	}

	.chat-text {
		margin: 0;
		white-space: pre-wrap;
		line-height: 1.45;
	}

	.chat-meta {
		margin: 0.35rem 0 0;
		font-size: 0.82rem;
		color: #475569;
	}

	.typing-bubble {
		display: flex;
		align-items: center;
		gap: 0.25rem;
	}

	.dot {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		background: #9ca3af;
		animation: bounce 1s infinite ease-in-out;
	}

	.dot:nth-child(2) {
		animation-delay: 0.15s;
	}

	.dot:nth-child(3) {
		animation-delay: 0.3s;
	}

	.answer-text {
		white-space: pre-wrap;
		line-height: 1.55;
	}

	.sources-toggle {
		justify-self: start;
	}

	@keyframes bounce {
		0%,
		80%,
		100% {
			transform: translateY(0);
			opacity: 0.7;
		}

		40% {
			transform: translateY(-3px);
			opacity: 1;
		}
	}

	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}

	@media (max-width: 920px) {
		.panel-grid {
			grid-template-columns: 1fr;
		}

		.topbar {
			flex-direction: column;
		}

		.topbar-right {
			justify-items: start;
		}
	}

	@media (max-width: 640px) {
		.app-shell {
			padding: 0.8rem;
		}

		.section-head {
			flex-direction: column;
			align-items: flex-start;
		}
	}
</style>