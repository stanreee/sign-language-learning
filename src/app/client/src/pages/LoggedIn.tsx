import asl from '../images/asl.png';
import { NavLink } from 'react-router-dom';

console.log(asl);

const Account = () => {

    return(
    <div className="App">
      <header className = "App-header">
        <img src={asl} alt="asl" />
        <h3>You have successfully logged into ASLingo!</h3>

        <div className = 'box'>
          <NavLink 
            to="/"
            style={({ isActive, isPending }) => {
              return {
                fontWeight: isActive ? "bold" : "bold",
                color: isPending ? "red" : '#003459',
              };
            }}
          >
            Return to Homepage
          </NavLink>
        </div>

      </header>
    </div>
    );
}

export default Account;