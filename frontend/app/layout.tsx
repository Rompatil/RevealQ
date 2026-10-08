import type { Metadata } from "next";
import "@fontsource-variable/bricolage-grotesque/opsz.css";
import "@fontsource-variable/public-sans/index.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "RevealQ",
  description: "Upload business data and see what changed, what's next, and what looks wrong.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
