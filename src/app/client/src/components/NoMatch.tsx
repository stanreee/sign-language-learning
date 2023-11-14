import { NavLink } from 'react-router-dom';

const NoMatch = () => {
	return (
		<div>
			<div className="container">
				<div className="error-page">
					<h1 className="error-code">404</h1>
					<p className="error-text">Page not found</p>
					{'\n Looks like something went wrong! Click below to return to the main page'}
					<NavLink 
						to="/"
						style={({ isActive, isPending }) => {
							return {
							fontWeight: isActive ? "bold" : "bold",
							color: isPending ? "red" : '#003459',
							};
						}} >
						Return to Home Page
					</NavLink>
				</div>
			</div>
		</div>
	);
}

export default NoMatch;