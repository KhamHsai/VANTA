import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "VANTA | Smarter waste decisions",
  description: "Real-time waste classification and disposal guidance.",
};

const navItems = [
  { href: "/", label: "Home" },
  { href: "/scan", label: "Scan" },
  { href: "/dashboard", label: "Dashboard" },
];

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <header className="border-b border-emerald-950/10 bg-[#f5f7f2]/90 backdrop-blur">
          <nav className="mx-auto flex max-w-6xl items-center justify-between px-5 py-5 sm:px-8" aria-label="Main navigation">
            <Link href="/" className="text-xl font-black tracking-[0.22em] text-emerald-950">
              VANTA
            </Link>
            <div className="flex items-center gap-5 text-sm font-semibold text-emerald-950/70 sm:gap-8">
              {navItems.map((item) => (
                <Link key={item.href} href={item.href} className="transition hover:text-emerald-700">
                  {item.label}
                </Link>
              ))}
            </div>
          </nav>
        </header>
        {children}
        <footer className="border-t border-emerald-950/10 px-5 py-8 text-center text-sm text-emerald-950/55">
          VANTA · Better sorting starts with better information.
        </footer>
      </body>
    </html>
  );
}

