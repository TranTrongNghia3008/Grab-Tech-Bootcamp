export function parseAISummary(aiSummary) {
    // Bỏ khung markdown ```html
    let rawHtml = aiSummary
      .replace(/^```html\n/, '')
      .replace(/```$/, '');
  
    // Tạo một DOM parser tạm thời nếu chạy trong trình duyệt
    const parser = new DOMParser();
    const doc = parser.parseFromString(rawHtml, "text/html");
  
    // Lấy phần body, hoặc toàn bộ nếu không có body
    const bodyContent = doc.body?.innerHTML || rawHtml;
  
    // Thêm Tailwind classes vào các tag
    return bodyContent
      .replace(/<div>/g, '<div class="space-y-4 text-sm text-gray-800">')
      .replace(/<ul>/g, '<ul class="list-disc pl-6 space-y-1">')
      .replace(/<li>/g, '<li class="text-gray-700">')
      .replace(/<p>/g, '<p class="text-gray-600">')
      .replace(/<strong>/g, '<strong class="text-gray-900">');
  }
  
  