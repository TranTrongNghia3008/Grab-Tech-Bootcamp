import { useEffect, useRef, useState } from "react";
import { Send, Plus, Menu } from "lucide-react";
import { useAppContext } from "../../contexts/AppContext";
import { getAllSessions, getSessionState, interactChatbot, startConversation } from "../../components/services/chatbotServices";

export default function ChatbotPanel() {
    const { state } = useAppContext(); 
    const { datasetId } = state;
    const [conversations, setConversations] = useState([]);
    const [currentConv, setCurrentConv] = useState(null);
    const [history, setHistory] = useState([]);
    const [message, setMessage] = useState("");
    const [loading, setLoading] = useState(false);
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const [editingConvId, setEditingConvId] = useState(null);
    const [editedName, setEditedName] = useState("");
    const chatEndRef = useRef();
    // const [sessionId, setSessionId] = useState(null);


    useEffect(() => {
        async function fetchConversations() {
            try {
            const res = await getAllSessions(datasetId);
            const sessions = res.sessions
            const formatted = sessions.map((s) => ({
                id: s.session_uuid,
                name: s.name || `💬 Chat ${s.session_uuid.slice(0, 6)}`,
                sessionId: s.session_uuid,
            }));

            setConversations(formatted);
            if (formatted.length > 0) {
                setCurrentConv(formatted[0]);
                // setSessionId(formatted[0].sessionId);
            }
            } catch (err) {
                console.error("Failed to fetch sessions", err);
            }
        }

        if (datasetId) fetchConversations();
    }, [datasetId]);


    useEffect(() => {
        async function fetchConversation(conv) {
            try {
                const res = await getSessionState(datasetId, conv.sessionId);

                const logs = res.journey_log;
                const history = [];

                for (let i = 0; i < logs.length; i++) {
                if (logs[i].event_type === "USER_QUERY") {
                    const userMsg = logs[i];
                    const nextResponse = logs[i + 1];

                    if (nextResponse && nextResponse.event_type === "TEXT_RESPONSE") {
                    history.push({
                        id: `msg-${i}`,
                        question: userMsg.payload.content,
                        answer: nextResponse.payload.content,
                        timestamp: new Date(userMsg.timestamp).toLocaleString(),
                    });
                    i++; // Skip next because already processed
                    }
                }
                }

                setHistory(history);
            } catch (err) {
                console.error("Failed to load conversation history", err);
                setHistory([]);
            }
        }
        if (currentConv) fetchConversation(currentConv)
    }, [currentConv]);

    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [history]);

    const handleSend = async () => {
        if (!message.trim() || !currentConv) return;

        const newMessage = {
            id: Date.now().toString(),
            question: message,
            answer: "Thinking..."
        };

        setHistory((prev) => [...prev, newMessage]);
        setMessage("");
        setLoading(true);

        try {
            const sessionId = currentConv.id
            const res = await interactChatbot(datasetId, sessionId , message);
            console.log("interactChatbot: ", res)
            const reply = res.responses?.[0]?.content || "No response";

            setHistory((prev) =>
            prev.map((msg) =>
                msg.id === newMessage.id ? { ...msg, answer: reply } : msg
            )
            );
        } catch (err) {
            console.error("Failed to get chatbot response", err);
            setHistory((prev) =>
            prev.map((msg) =>
                msg.id === newMessage.id ? { ...msg, answer: "Error getting response." } : msg
            )
            );
        } finally {
            setLoading(false);
        }
    };

    const handleNewConversation = async () => {
        try {
            const res = await startConversation(datasetId);
            const newConv = {
                id: res.session_id,
                name: `💬 New Chat ${conversations.length + 1}`,
                sessionId: res.session_id,
            };

            setConversations((prev) => [...prev, newConv]);
            setCurrentConv(newConv);
            // setSessionId(res.session_id);
            setHistory([]);
        } catch (err) {
            console.error("Failed to start new conversation", err);
        }
    };

    // const handleClickConversation = async (conv) => {
    //     setCurrentConv(conv);

    //     try {
    //         const res = await getSessionState(datasetId, conv.sessionId);

    //         const logs = res.journey_log;
    //         const history = [];

    //         for (let i = 0; i < logs.length; i++) {
    //         if (logs[i].event_type === "USER_QUERY") {
    //             const userMsg = logs[i];
    //             const nextResponse = logs[i + 1];

    //             if (nextResponse && nextResponse.event_type === "TEXT_RESPONSE") {
    //             history.push({
    //                 id: `msg-${i}`,
    //                 question: userMsg.payload.content,
    //                 answer: nextResponse.payload.content,
    //                 timestamp: new Date(userMsg.timestamp).toLocaleString(),
    //             });
    //             i++; // Skip next because already processed
    //             }
    //         }
    //         }

    //         setHistory(history);
    //     } catch (err) {
    //         console.error("Failed to load conversation history", err);
    //         setHistory([]);
    //     }
    // };


    const handleDeleteConversation = (id) => {
        setConversations((prev) => prev.filter((c) => c.id !== id));
        if (currentConv?.id === id) {
        setCurrentConv(null);
        setHistory([]);
        }
    };

    const handleDeleteMessage = (id) => {
        setHistory((prev) => prev.filter((msg) => msg.id !== id));
    };

    return (
        <div className="flex h-[630px] rounded-xl shadow-lg border overflow-hidden bg-white">

        {/* Sidebar */}
        {sidebarOpen && (
            <div className="w-64 bg-gray-100 border-r p-4 space-y-4 transition-all duration-300">
            <div className="flex justify-between items-center mb-2">
                <h3 className="text-sm font-bold text-gray-700">Conversations</h3>
                <button
                onClick={handleNewConversation}
                className="text-green-600 hover:cursor-pointer"
                title="New Chat"
                >
                <Plus size={18} />
                </button>
            </div>
            <ul className="space-y-1 text-sm">
                {conversations.map((conv) => (
                    <li
                    key={conv.id}
                    className={`flex justify-between items-center px-3 py-2 rounded group cursor-pointer ${
                        currentConv?.id === conv.id
                        ? "bg-green-100 text-green-600 font-semibold"
                        : "hover:bg-green-200 text-gray-800"
                    }`}
                    >
                    {editingConvId === conv.id ? (
                        <div className="flex items-center w-full gap-1">
                        <input
                            value={editedName}
                            onChange={(e) => setEditedName(e.target.value)}
                            className="text-sm flex-1 px-2 py-1 rounded border border-gray-300"
                        />
                        <button
                            onClick={() => {
                            setConversations((prev) =>
                                prev.map((c) =>
                                c.id === conv.id ? { ...c, name: editedName } : c
                                )
                            );
                            setEditingConvId(null);
                            }}
                            className="text-green-600 hover:text-green-800 text-xs"
                            title="Save"
                        >
                            ✅
                        </button>
                        <button
                            onClick={() => setEditingConvId(null)}
                            className="text-gray-400 hover:text-gray-600 text-xs"
                            title="Cancel"
                        >
                            ❌
                        </button>
                        </div>
                    ) : (
                        <>
                        <span
                            onClick={() => setCurrentConv(conv)}
                            className="truncate max-w-[120px] flex-1"
                        >
                            {conv.name}
                        </span>

                        <div className="flex gap-1 opacity-0 group-hover:opacity-100">
                            <button
                            onClick={() => {
                                setEditingConvId(conv.id);
                                setEditedName(conv.name);
                            }}
                            className="text-yellow-500 hover:text-yellow-600 text-xs"
                            title="Rename"
                            >
                            ✏️
                            </button>
                            <button
                            onClick={() => handleDeleteConversation(conv.id)}
                            className="text-red-400 hover:text-red-600 text-xs"
                            title="Delete"
                            >
                            ✕
                            </button>
                        </div>
                        </>
                    )}
                    </li>
                ))}
            </ul>

            </div>
        )}

        {/* Chat Area */}
        <div className="flex-1 flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-2 border-b bg-gray-50">
            <div className="flex items-center gap-2">
                <button
                onClick={() => setSidebarOpen((prev) => !prev)}
                className="text-gray-600 hover:text-gray-800"
                title="Toggle Conversations"
                >
                <Menu size={20} />
                </button>
                <h2 className="font-semibold text-gray-700 text-sm">
                {currentConv?.name || "No conversation selected"}
                </h2>
            </div>
            </div>

            {/* Chat Messages */}
            <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4 bg-white">
            {history.map((msg) => (
                <div key={msg.id} className="space-y-2">
                    {/* User */}
                    <div className="flex items-start gap-3">
                        <img
                            src="/avatars/user.png" // ảnh user
                            alt="User"
                            className="w-7 h-7 rounded-full object-cover border border-gray-300"
                        />
                        <div className="bg-green-50 px-4 py-2 rounded-md text-sm flex-1 shadow-sm">
                            <div className="text-xs text-gray-400 mb-1">
                                {msg.timestamp}
                            </div>
                            <div className="flex justify-between items-start">
                            <p className="text-green-600 font-medium">{msg.question}</p>
                            <button
                                onClick={() => handleDeleteMessage(msg.id)}
                                className="text-xs text-red-400 hover:text-red-600 ml-2"
                                title="Delete message"
                            >
                                ✕
                            </button>
                            </div>
                        </div>
                    </div>

                    {/* Bot */}
                    <div className="flex items-start gap-3">
                    <img
                        src="/avatars/bot.png" // ảnh bot
                        alt="Bot"
                        className="w-7 h-7 rounded-full object-cover border border-gray-300"
                    />
                    <div className="bg-gray-100 px-4 py-2 rounded-md text-sm text-gray-800 flex-1 shadow-sm">
                        <div className="text-xs text-gray-400 mb-1">
                            {msg.timestamp}
                        </div>
                        {msg.answer}
                    </div>
                    </div>
                </div>
            ))}

            <div ref={chatEndRef}></div>
            </div>

            {/* Input Area */}
            <div className="border-t px-4 py-3 bg-gray-50 flex items-center gap-2">
            <input
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSend()}
                type="text"
                placeholder="Ask your dataset anything..."
                className="flex-1 border border-gray-300 px-4 py-2 rounded-md focus:outline-none focus:ring-2 focus:ring-green-600 text-sm"
            />
            <button
                onClick={handleSend}
                disabled={loading}
                className="text-green-600 hover:text-green-800 disabled:opacity-40"
                title="Send"
            >
                <Send size={20} />
            </button>
            </div>
        </div>
        </div>
    );
}