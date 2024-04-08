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
							<NavLink reloadDocument to="/">Home</NavLink>
						</li>
						<li>
							<NavLink reloadDocument to="/learn">Learn</NavLink>
						</li>
						<li>
							<NavLink reloadDocument to="/practice">Practice</NavLink>
						</li>
						<li>
							<NavLink reloadDocument to="/exercises">Exercises</NavLink>
						</li>
					</ul>
				</div>
			</nav>
		</div>
	);
};

export default NavBar;