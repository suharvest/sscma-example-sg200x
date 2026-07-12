import ReactDOM from "react-dom/client";
import App from "./App.tsx";
import "@/i18n";
import "antd-mobile/es/global";
// latin subsets only — CJK text (e.g. name_zh) falls back to system fonts
import "@fontsource/montserrat/latin-400.css";
import "@fontsource/montserrat/latin-500.css";
import "@fontsource/montserrat/latin-600.css";
import "@fontsource/montserrat/latin-700.css";
import "@/assets/style/index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
