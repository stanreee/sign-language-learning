import { NavLink } from 'react-router-dom';


const NavBar = () => {
    return (
		<div>
			<nav>
				<div className="nav-items container">
					<div className="logo">
						<a href="/">
							<h1>ASLingo</h1>
						</a>
					</div>
                    <ul>
						<li>
							<NavLink to="/">Home</NavLink>
						</li>
						<li>
							<NavLink to="/learn">Learn</NavLink>
						</li>
						<li>
							<NavLink to="/exercises">Exercises</NavLink>
						</li>
						<li>
							<NavLink to="/practice">Practice</NavLink>
						</li>
						<li>
							<NavLink to="/account">Login</NavLink>
						</li>
					</ul>
				</div>
			</nav>
		</div>
	);
};

export default NavBar;