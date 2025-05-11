export function formatHistoryFromLogs(logs) {
    console.log("logs: ", logs)
  const history = [];

  for (let i = 0; i < logs.length; i++) {
    const log = logs[i];

    if (log.event_type === "USER_QUERY") {
      const question = log.payload?.content || "";
      const timestamp = new Date(log.timestamp).toLocaleString();
      const answerParts = [];

      for (let j = i + 1; j < logs.length; j++) {
        const next = logs[j];

        if (next.event_type === "USER_QUERY") break;

        if (next.event_type === "TEXT_RESPONSE") {
          const text = next.payload?.content;
          if (text) {
            answerParts.push(`<p class="text-gray-800">${text}</p>`);
          }
        }

        if (next.event_type === "CALCULATION_RESULT") {
          const output = next.payload?.output || next.payload?.message;
          if (output) {
            answerParts.push(`
              <div class="bg-yellow-50 border border-yellow-200 rounded-md p-2 text-sm text-yellow-800 whitespace-pre-wrap">
                <strong>Calculation Result:</strong><br/>${output}
              </div>`);
          }
        }

        if (next.event_type === "PLOT_GENERATED") {
            const imgPath = next.payload?.path?.replace(/\\/g, "/").replace("../FE/automation-data-analysts/public", "");
            const actions = next.payload?.next_actions || [];

            let actionsHTML = "";
            if (actions.length > 0) {
                actionsHTML = `
                <div class="mt-2 flex flex-wrap gap-2">
                    ${actions.map((action) => `
                    <span class=" text-gray-400">You can ask more (click here): </span>
                    <button 
                        class="bg-green-100 hover:bg-green-200 text-green-800 px-3 py-1 text-xs rounded cursor-pointer suggestion-btn"
                        data-suggestion="${action}"
                    >${action}</button>
                    `).join("")}
                </div>`;
            }

            if (imgPath) {
                answerParts.push(`
                <div class="mt-2">
                    <strong>Generated Plot:</strong>
                    <img src="${imgPath}" alt="Plot" class="mt-1 rounded shadow max-w-full border" />
                    ${actionsHTML}
                </div>`);
            }
            }


        if (next.event_type === "PLOT_ANALYSIS_RESULT") {
          const insights = next.payload?.insights;
          if (insights) {
            answerParts.push(`
              <div class="bg-blue-50 border border-blue-200 rounded-md p-3 text-sm text-blue-800 whitespace-pre-wrap">
                <strong>Plot Analysis:</strong><br/>${insights}
              </div>`);
          }
        }

        i = j; // Đánh dấu đã xử lý phản hồi
      }

      history.push({
        id: `msg-${i}`,
        question,
        answer: answerParts.join("\n"),
        timestamp,
        type: "html",
      });
    }
  }

  return history;
}
