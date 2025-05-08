// src/services/apiClient.js
const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function apiClient(path, { method = "GET", body, headers = {}, ...customConfig } = {}) {
    const token = localStorage.getItem("token");
  
    const isFormData = body instanceof FormData;
  
    const config = {
      method,
      headers: {
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(isFormData ? {} : { "Content-Type": "application/json" }),
        ...headers,
      },
      ...customConfig,
    };
  
    if (body && method !== "GET" && method !== "HEAD") {
      config.body = isFormData ? body : JSON.stringify(body);
    }
  
    try {
      const response = await fetch(`${BASE_URL}${path}`, config);
  
      if (!response.ok) {
        const errorBody = await response.json();
        throw new Error(errorBody.message || "API error");
      }
  
      if (response.status === 204) return null;
  
      return await response.json();
    } catch (error) {
      console.error("API call error:", error);
      throw error;
    }
  }
  
export default apiClient;
