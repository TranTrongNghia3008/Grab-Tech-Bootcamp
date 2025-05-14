export function parseAISummary(aiSummary) {
  // Tìm tất cả các khối ```html ... ``` bằng regex (multiline)
  const htmlBlocks = [...aiSummary.matchAll(/```html\s*([\s\S]*?)\s*```/g)];

  // Nếu không có khối markdown nào, trả về raw
  if (htmlBlocks.length === 0) return aiSummary;

  return htmlBlocks
    .map(([, rawHtml]) => {
      // Dùng DOMParser để parse đoạn HTML
      const parser = new DOMParser();
      const doc = parser.parseFromString(rawHtml, "text/html");
      const content = doc.body?.innerHTML || rawHtml;

      // Thêm Tailwind class
      return content
        .replace(/<div>/g, '<div class="space-y-2 text-sm text-gray-800">')
        .replace(/<ul>/g, '<ul class="list-disc pl-4">')
        .replace(/<li>/g, '<li class="text-gray-700">')
        .replace(/<p>/g, '<p class="text-gray-600">')
        .replace(/<strong>/g, '<strong class="text-gray-900">');
    })
}