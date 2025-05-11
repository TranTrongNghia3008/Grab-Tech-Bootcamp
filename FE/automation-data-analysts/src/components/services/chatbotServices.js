import apiClient from "./apiClient";

export function startConversation(datasetId) {
    return apiClient(`/sessions/start/${datasetId}`, {
        method: "POST",
    });
}

export function getSessionState(datasetId, sessionId) {
    return apiClient(`/sessions/${datasetId}/state?session_id=${sessionId}`, {
        method: "GET",
    })
}

export function interactChatbot(datasetId, sessionId, query) {
    const payload = {
        "query": query
    }
    return apiClient(`/interact/${datasetId}?session_id=${sessionId}`, {
        method: "POST",
        body: payload
    })
}

export function getAllSessions(datasetId) {
    return apiClient(`/sessions/list/${datasetId}`, {
        method: "GET",
    });
}

export function getKStateLatest(datasetId, k, sessionId) {
    return apiClient(`/sessions/${datasetId}/state/latest?k=${k}&session_id=${sessionId}`, {
        method: "GET"
    })
}

export function clearHistory(datasetId, sessionId) {
    return apiClient(`/sessions/${datasetId}/clear-history?session_id=${sessionId}`, {
        method: "POST"
    })
}