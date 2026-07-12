/**
 * Supervisor API response code helpers. The backend returns `code` as a
 * number or a string depending on the endpoint, so both are accepted.
 */

interface ICodeCarrier {
  code?: number | string;
}

/** code 0 = success. */
export const isOk = (res: ICodeCarrier): boolean =>
  res.code === 0 || res.code === "0";

/** code -2 = busy (another operation holds the lock / device busy). */
export const isBusy = (res: ICodeCarrier): boolean =>
  res.code === -2 || res.code === "-2";
