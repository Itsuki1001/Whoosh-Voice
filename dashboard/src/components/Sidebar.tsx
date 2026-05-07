import { NavLink } from "react-router-dom";
import { Home, MessageCircle, Clock, Settings, Palmtree, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

const nav = [
  { to: "/", icon: Home, label: "Home", end: true },
  { to: "/conversations", icon: MessageCircle, label: "Conversations" },
  { to: "/follow-ups", icon: Clock, label: "Follow Ups" },
  { to: "/settings", icon: Settings, label: "Settings" },
];

export const Sidebar = () => {
  return (
    <aside className="flex h-screen w-60 flex-col border-r border-sidebar-border bg-sidebar">
      <div className="flex items-center gap-2 px-5 py-5">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent">
          <Palmtree className="h-5 w-5 text-primary" />
        </div>
        <div className="leading-tight">
          <div className="text-sm font-semibold">Ocean View</div>
          <div className="text-sm font-semibold">Resort</div>
        </div>
      </div>

      <nav className="mt-2 flex-1 space-y-1 px-3">
        {nav.map(({ to, icon: Icon, label, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
                isActive
                  ? "bg-sidebar-active text-sidebar-active-foreground"
                  : "text-sidebar-foreground hover:bg-secondary"
              )
            }
          >
            <Icon className="h-4 w-4" />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="m-3 flex items-center gap-2 rounded-lg border border-sidebar-border bg-card p-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-accent text-xs font-semibold text-accent-foreground">
          AR
        </div>
        <div className="flex-1 leading-tight">
          <div className="text-sm font-medium">Ocean View Resort</div>
          <div className="text-xs text-muted-foreground">Admin</div>
        </div>
        <ChevronDown className="h-4 w-4 text-muted-foreground" />
      </div>
    </aside>
  );
};
