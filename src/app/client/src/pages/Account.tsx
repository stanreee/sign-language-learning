import { NavLink } from 'react-router-dom';
import asl from '../images/asl.png';

console.log(asl);

const Account = () => {
    return(
    <div className="App">
      <header className = "App-header">
        <img src={asl} alt="asl" />
        <h1>Welcome to ASLingo!</h1>

        <div className = 'box'>
          <NavLink 
            to="/login"
            style={({ isActive, isPending }) => {
              return {
                fontWeight: isActive ? "bold" : "bold",
                color: isPending ? "red" : '#003459',
              };
            }}
          >
            Login
          </NavLink>

            <h3> or </h3>

          <NavLink 
            to="/signup"
            style={({ isActive, isPending }) => {
              return {
                fontWeight: isActive ? "bold" : "bold",
                color: isPending ? "red" : '#003459',
              };
            }}
          >
            Sign Up
          </NavLink>
        </div>
      </header>
    </div>
    );
}

export default Account;