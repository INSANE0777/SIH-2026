import { Chakra_Petch, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const chakra = Chakra_Petch({
  subsets: ["latin"],
  weight: ["500", "600", "700"],
  variable: "--font-disp",
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "700"],
  variable: "--font-mono",
});

export const metadata = {
  title: "SwaraKavach — Voice Integrity Console",
  description:
    "Real-time voice-clone impersonation detection & prevention — SIH 2026 PS26104",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={`${chakra.variable} ${mono.variable}`}>{children}</body>
    </html>
  );
}
