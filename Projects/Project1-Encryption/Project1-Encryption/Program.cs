using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
	class Program
	{
		static void Main(string[] args)
		{
			string inputText, encryptedText, decryptedText;

			System.Console.Write("Enter the text you would like to encrypt: ");
			inputText = System.Console.ReadLine();
			encryptedText = Encrypt(inputText);
			System.Console.WriteLine("Encrypted text:" + encryptedText);
			decryptedText = Decrypt(encryptedText);
			System.Console.WriteLine("Decrypted text: " + decryptedText);
			System.Console.Write("Press Enter to Quit.");
			inputText = System.Console.ReadLine();
		} // End of Main


		static string Encrypt(string toEncrypt)
		{
			// Encrypts the input by applying a Caesar cipher of 26-character
			// It then salts the encryption by adding 1-26 random characters in between
			// each encrypted character. It stores the key of random characters at the
			// end of the encrypted string as a character.

			string encrypted = "";
			char encryptedChar = (char)(0);
			Random encRand = new Random();
			Random saltRand = new Random();
			Random lowerUpper = new Random();
			//int caseAddition; // Used for character-only salt
			int encryptionKey = encRand.Next(1, 26);
			foreach (char c in toEncrypt) // iterate through string to encrypt
			{
				string saltString = "";
				for (int x = 0; x <= encryptionKey; x++)
				{
					// ------------------------------------------
					// Uncomment for only characters as salt
					//int caseRand = lowerUpper.Next(0, 1);
					//int salt = saltRand.Next(0, 25);
					//caseAddition = caseRand == 1 ? 97 : 65;
					//saltString += (char)(salt + caseAddition);
					//------------------------------------------
					int salt = saltRand.Next(33, 165);
					saltString += (char)(salt);
				} // Salt string loop
				if (c > 90)
				{
					encryptedChar = (char)(219 - c);
				} // check for lower case
				else
				{
					encryptedChar = (char)(155 - c);
				} // check for upper case
				encrypted += (encryptedChar + saltString);
			} // toEncrypt loop
			encrypted += (char)(encryptionKey + 64);
			System.Console.WriteLine("Encryption key (int): " + encryptionKey);
			System.Console.WriteLine("Encryption key (char): " + (char)(encryptionKey+64));
			return encrypted;
		} // End Encrypt function


		static string Decrypt(string toDecrypt)
		{
			// Takes the encrypted text from the Encryption function, stores the
			// final character as an interger, strips the character from the 
			// string and then iterates through the encrypted string, stepping by
			// that interger value. Each step, it deciphers the caesar cipher
			// using the same 26-char method.

			string decrypted = "";
			char decryptedChar;
			int encKey = toDecrypt[toDecrypt.Length - 1] - 62;
			toDecrypt = toDecrypt.Remove(toDecrypt.Length - 1);
			for (int i = 0; i < toDecrypt.Length; i += encKey)
			{
				if (toDecrypt[i] > 90)
				{
					decryptedChar = (char)(219 - toDecrypt[i]);
				} // check for lower case & decrypt
				else
				{
					decryptedChar = (char)(155 - toDecrypt[i]);
				} // decrypt upper case
				decrypted += decryptedChar;
			} // Iteration through string to encrypt
			return decrypted;
		} // End Decrypt function
	}
}
