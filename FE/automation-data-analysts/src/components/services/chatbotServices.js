import apiClient from "./apiClient";

export function startConversation(datasetId) {
    return apiClient(`/sessions/start/${datasetId}`, {
        method: "POST",
    });
}

export function interactChatbot(datasetId, sessionId, query) {
    // const formData = new FormData();
    // formData.append("query", query);
    const payload = {
        "query": query
    }
    return apiClient(`/interact/${datasetId}?session_id=${sessionId}`, {
        method: "POST",
        body: payload
    })
}