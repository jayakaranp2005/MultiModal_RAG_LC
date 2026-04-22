import { API_BASE } from './config';

const DEFAULT_TIMEOUT_MS = 60000;

function buildUrl(path) {
	return `${API_BASE}${path}`;
}

function withTimeout(timeoutMs = DEFAULT_TIMEOUT_MS) {
	const controller = new AbortController();
	const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
	return { controller, timeoutId };
}

async function parseResponse(response, fallbackMessage) {
	let payload = null;

	try {
		payload = await response.json();
	} catch (_error) {
		payload = null;
	}

	if (!response.ok) {
		const message =
			(payload && (payload.detail || payload.error || payload.message)) ||
			`${fallbackMessage} (HTTP ${response.status})`;
		throw new Error(message);
	}

	if (!payload || typeof payload !== 'object') {
		throw new Error('Server returned an unexpected response format.');
	}

	return payload;
}

async function fetchJson(path, options = {}, fallbackMessage = 'Request failed', timeoutMs = DEFAULT_TIMEOUT_MS) {
	const { controller, timeoutId } = withTimeout(timeoutMs);

	try {
		const response = await fetch(buildUrl(path), {
			...options,
			signal: controller.signal
		});
		return await parseResponse(response, fallbackMessage);
	} catch (error) {
		if (error.name === 'AbortError') {
			throw new Error('Request timed out. Please try again.');
		}
		throw error instanceof Error ? error : new Error(fallbackMessage);
	} finally {
		clearTimeout(timeoutId);
	}
}

export async function health() {
	return fetchJson('/health', { method: 'GET' }, 'Unable to check backend health.', 15000);
}

export async function getIndexed() {
	return fetchJson('/indexed', { method: 'GET' }, 'Unable to fetch indexed PDFs.', 30000);
}

export async function uploadPdf(file, reindex) {
	const formData = new FormData();
	formData.append('file', file);

	const query = new URLSearchParams({ reindex: String(Boolean(reindex)) });
	return fetchJson(
		`/upload_pdf?${query.toString()}`,
		{
			method: 'POST',
			body: formData
		},
		'Failed to upload and index the PDF.',
		60000
	);
}

export async function ask(question) {
	return fetchJson(
		'/ask',
		{
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({ question })
		},
		'Failed to get an answer from the backend.',
		60000
	);
}
