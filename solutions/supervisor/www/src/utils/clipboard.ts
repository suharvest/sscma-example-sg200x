import { message } from "antd";

/**
 * Copy text to the clipboard and toast the outcome. Messages are passed in
 * already translated (components call t() at the call site).
 */
export function copyText(text: string, successMsg: string, failMsg: string) {
  navigator.clipboard
    ?.writeText(text)
    .then(() => message.success(successMsg))
    .catch(() => message.error(failMsg));
}
