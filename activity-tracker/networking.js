const BASE_URL = "http://192.168.1.5:5000"

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

const registerMeasurement = (data) => {
	const requestOptions = {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({...data}),
		timeout: 500
	};

	return fetch(BASE_URL + "/measurements", requestOptions)
		.then(res => res.json())
}

const registerCalibration = (data, type) => {
	const requestOptions = {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({...data, type}),
		timeout: 500
	};

	return fetch(BASE_URL + "/tuning/submit", requestOptions)
		.then(res => res.json())
}

const resetCalibration = () => {
	const requestOptions = {
		method: 'DELETE',
		headers: { 'Content-Type': 'application/json' },
		timeout: 500
	};

	return fetch(BASE_URL + "/tuning/submit", requestOptions)
		.then(res => res.json())
}

const startCalibration = () => {
	const requestOptions = {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		timeout: 500
	};

	return fetch(BASE_URL + "/tuning/start", requestOptions)
		.then(res => res.json())
}

const resetModel = () => {
	const requestOptions = {
		method: 'DELETE',
		headers: { 'Content-Type': 'application/json' },
		timeout: 500
	};

	return fetch(BASE_URL + "/tuning/start", requestOptions)
		.then(res => res.json())
}

export { registerMeasurement, registerCalibration, resetCalibration, startCalibration, resetModel }
