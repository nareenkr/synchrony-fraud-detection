import { NavLink, Outlet } from "react-router-dom";
import { authRequired, clearAccessKey } from "../api";

export function Shell() {
  function signOut() {
    clearAccessKey();
    window.location.assign("/");
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <NavLink to="/" className="brand" aria-label="Synchrony home">
        <span className="brand-mark" aria-hidden="true">S</span>
        <span>Synchro<span>ny</span></span>
      </NavLink>
      <nav aria-label="Primary navigation">
        <NavLink to="/" end><span aria-hidden="true">◇</span> Monitoring</NavLink>
        <NavLink to="/model-info"><span aria-hidden="true">◎</span> Model info</NavLink>
      </nav>
      <div className="sidebar-note"><span className="status-dot" />System ready<p>Local prototype</p></div>
      {authRequired && <button className="sign-out" type="button" onClick={signOut}>Clear access key</button>}
    </aside>
    <div className="page-frame">
      <header className="topbar">
        <div><p className="eyebrow">Decision intelligence</p><strong>Digital lending protection</strong></div>
        <div className="prototype-pill">Synthetic data · Prototype</div>
      </header>
      <main><Outlet /></main>
      <footer>This prototype provides decision support only. It is not validated for autonomous credit or fraud decisions.</footer>
    </div>
  </div>;
}
