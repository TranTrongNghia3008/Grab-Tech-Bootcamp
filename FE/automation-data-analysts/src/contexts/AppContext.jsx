import React, { createContext, useContext, useState } from "react";

const AppContext = createContext();

export function useAppContext() {
  return useContext(AppContext);
}

export function AppProvider({ children }) {
  const [state, setState] = useState({
    datasetId: null,
    sessionId: null,
    comparisonResults: {},
    tuningResults: null,
    predictedResults: null

    // ... thêm nữa nếu cần
  });

  // Hàm cập nhật từng phần
  const updateState = (updates) => {
    setState((prev) => ({ ...prev, ...updates }));
  };

  return (
    <AppContext.Provider value={{ state, updateState }}>
      {children}
    </AppContext.Provider>
  );
}
