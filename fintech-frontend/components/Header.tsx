"use client";

import Link from "next/link";
import ThemeController from "./ThemeController";
import Dropdown from "./Dropdown";

export default function Header() {
  return (
    <header className="w-full bg-white shadow-md">
      <div className="flex items-center justify-between mx-auto px-4 py-4">
        
        <Link href="/" className="text-2xl font-bold text-blue-600">
        Fintech
        </Link>

        {/* Navigation links */}
        <nav className="flex space-x-8 text-lg text-gray-700">
            <Dropdown/>
         <ThemeController/>
        </nav>

      </div>
    </header>
  );
}
