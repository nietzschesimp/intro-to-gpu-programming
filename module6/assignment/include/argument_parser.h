#ifndef ARGUMENT_PARSER_H_
#define ARGUMENT_PARSER_H_

#include <algorithm>
#include <vector>
#include <string>



class ArgumentParser {
	public:
		ArgumentParser(int& argc, char** argv) {
			for (int i = 1; i < argc; ++i) {
				this->arg_list.push_back(std::string(argv[i]));
			}
		}

		inline const std::string& get_option(const std::string& opt) const {
			std::vector<std::string>::const_iterator iter;
			iter = std::find(arg_list.begin(), arg_list.end(), opt);
			if (iter != arg_list.end() && ++iter != arg_list.end()) {
				return *iter;
			}
			static const std::string empty("");
			return empty;
		}

		inline bool exists(const std::string& opt) const {
			return std::find(arg_list.begin(), arg_list.end(), opt) != arg_list.end();
		}

	private:
		std::vector<std::string> arg_list;
};

#endif
