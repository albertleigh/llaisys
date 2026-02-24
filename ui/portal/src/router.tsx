/**
 * router.tsx — React Router configuration.
 *
 * Routes:
 *  /          → ChatView (default — new or active conversation)
 */

import { createBrowserRouter } from "react-router-dom";
import { AppLayout } from "./components";
import { ChatView } from "./components";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppLayout />,
    children: [
      {
        index: true,
        element: <ChatView />,
      },
    ],
  },
]);
