function getSizeByNumber(maxNum, start = 0, gap = 1) {
  const sizeObj = {};
  for (let index = start; index <= maxNum; index += gap) {
    sizeObj[index] = index;
  }
  return sizeObj;
}

export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // Seeed showcase design tokens
        primary: "#8fc31f",
        accent: "#8fc31f",
        "accent-hover": "#81ae1b",
        fg: "#0a0a0a",
        muted: "#666666",
        line: "#e6e6e6",
        surface: "#f7f7f7",
        "3d": "#3d3d3d",
        background: "#f7f7f7",
        selected: "#eef3fd",
        disable: "#c8d4ee",
        text: "#878B7E",
        error: "#D54941",
      },
      fontFamily: {
        display: [
          "Montserrat",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "SFMono-Regular",
          "ui-monospace",
          "Consolas",
          "Liberation Mono",
          "Menlo",
          "Courier",
          "monospace",
        ],
      },
      width: getSizeByNumber(750),
      minWidth: getSizeByNumber(300),
      height: getSizeByNumber(100),
      fontSize: getSizeByNumber(50, 12),
      opacity: getSizeByNumber(1, 0, 0.5),
      borderRadius: getSizeByNumber(25),
      zIndex: getSizeByNumber(100, 0, 10),
      //  -20：处理负数失效问题（-mt-20、-top-20）
      spacing: getSizeByNumber(200, -60),
    },
  },
  plugins: [],
};
