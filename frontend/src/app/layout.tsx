import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/layout/Sidebar";
import { Header } from "@/components/layout/Header";

export const metadata: Metadata = {
  title: "PlantIQ — Manufacturing Intelligence",
  description: "Adaptive Multi-Objective Optimization of Industrial Batch Processes",
  icons: {
    icon: "/favicon.ico",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <div className="flex h-screen bg-slate-50 overflow-hidden">
          <Sidebar />
          <div className="flex-1 flex flex-col overflow-hidden">
            <Header />
            <main className="flex-1 overflow-y-auto custom-scrollbar bg-slate-50/50">
              {children}
            </main>
          </div>
        </div>
      </body>
    </html>
  );
}
