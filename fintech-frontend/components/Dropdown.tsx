export default function Dropdown() {
  return (
    <div className="dropdown dropdown-end">
      {/* Avatar trigger */}
      <div
        tabIndex={0}
        role="button"
        className="btn btn-ghost btn-circle avatar"
      >
        <div className="w-10 rounded-full">
          <img
            alt="User avatar"
            src="https://img.daisyui.com/images/stock/photo-1534528741775-53994a69daeb.webp"
          />
        </div>
      </div>
      <ul
        tabIndex={-1}
        className="menu menu-sm dropdown-content bg-base-100 rounded-box z-10 mt-3 w-52 p-2 shadow"
      >
        <li>
          <a>Sign-in</a>
        </li>
        <li>
          <a>Sign-up</a>
        </li>
      </ul>
    </div>
  );
}
