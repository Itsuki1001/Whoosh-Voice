import { Sidebar } from "./Sidebar";
import { ReactNode } from "react";

export const DashboardLayout = ({ children }: { children: ReactNode }) => (
  <div className="flex min-h-screen bg-background">
    <Sidebar />
    <main className="flex-1 overflow-x-hidden p-6 lg:p-8">{children}</main>
  </div>
);
