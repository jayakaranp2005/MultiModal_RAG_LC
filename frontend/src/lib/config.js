const DEFAULT_API_BASE = 'http://127.0.0.1:8000';

function getStoredApiBase() {
	try {
		const value = localStorage.getItem('API_BASE');
		if (!value) {
			return null;
		}

		const trimmed = value.trim();
		return trimmed.length > 0 ? trimmed : null;
	} catch (_error) {
		return null;
	}
}

export const API_BASE = getStoredApiBase() || DEFAULT_API_BASE;