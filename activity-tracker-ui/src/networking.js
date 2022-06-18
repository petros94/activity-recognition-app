const BASE_URL = "http://localhost:5000"

async function fetchWithTimeout(resource, options = {}) {
	const { timeout = 8000 } = options;

	const controller = new AbortController();
	const id = setTimeout(() => controller.abort(), timeout);
	const response = await fetch(resource, {
		...options,
		signal: controller.signal
	});
	clearTimeout(id);
	return response;
}

const getMeasurement = (data) => {
	const requestOptions = {
		method: 'GET',
		headers: { 'Content-Type': 'application/json' },
		timeout: 500
	};

	return fetch(BASE_URL + "/measurements", requestOptions)
		.then(res => res.json())
}

export { getMeasurement }
