import React from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "../../dahs_dashboard.jsx";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
